/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "backend/graph_compiler/backend.h"

#include <algorithm>
#include <vector>
#include <map>

#include "frontend/parallel/context.h"
#include "backend/graph_compiler/transform.h"
#include "backend/common/session/session_factory.h"
#include "runtime/op_builder/op_lazy_builder.h"
#include "backend/common/optimizer/helper.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/parse/data_converter.h"
#include "ir/anf.h"
#include "pybind_api/ir/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "utils/callbacks.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "utils/scoped_long_running.h"
#ifdef ENABLE_D
#include "utils/callbacks_ge.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

namespace mindspore {
namespace compile {
bool Backend::GetCond(const BaseRef &c, bool *const value) {
  mindspore::ScopedLongRunning long_running;
  return BaseRefToBool(c, value);
}
bool Backend::GetIndex(const BaseRef &c, int64_t *const value) { return BaseRefToInt(utils::cast<ValuePtr>(c), value); }

Backend::Backend(const std::string &name) : name_(name) {
  MS_LOG(DEBUG) << "Select backend:" << name;
  convert_fn_ = MsVmConvert;
  is_multi_graph_sink_ = false;
}

LinConvertResult MsBackend::MsConvert(const GraphSegmentPtr &segment, const std::string &target) {
  MS_LOG(DEBUG) << "MsConvert";
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  LinConvertResult result;
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(segment->nodes_);
  result.inputs = inputs;
  result.outputs = outputs;
  result.graph_id = kInvalidGraphId;
  auto current_session = target_sess_;
  if (target != target_device_ && !target.empty()) {
    CreateOtherSession(target);
    current_session = other_sess_;
  }
  MS_EXCEPTION_IF_NULL(current_session);
  GraphId graph_id = current_session->CompileGraph(segment, outputs);
  segment->graph_id_ = graph_id;
  auto graph = current_session->GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &pre_segment : segment->pre_segments_) {
    MS_EXCEPTION_IF_NULL(pre_segment);
    MS_EXCEPTION_IF_NULL(target_sess_);
    auto pre_graph = target_sess_->GetGraph(pre_segment->graph_id_);
    if (pre_graph == nullptr) {
      MS_EXCEPTION_IF_NULL(other_sess_);
      pre_graph = other_sess_->GetGraph(pre_segment->graph_id_);
    }
    MS_EXCEPTION_IF_NULL(pre_graph);
    pre_graph->AddPostGraph(graph);
    graph->AddPreGraph(pre_graph);
    MS_LOG(INFO) << "Link graph " << pre_segment->graph_id_ << " to " << graph_id;
  }

  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return result;
  }
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (!pynative_mode || target != "Ascend") {
    if (target != target_device_ && !target.empty()) {
      MS_EXCEPTION_IF_NULL(other_sess_);
      other_sess_->BuildGraph(graph_id);
    } else if (!is_multi_graph_sink_) {
      MS_EXCEPTION_IF_NULL(target_sess_);
      target_sess_->BuildGraph(graph_id);
    }
  }
  result.run = std::make_shared<RunFunc>(
    [graph_id, target, this](const VectorRef &args) -> VectorRef { return MsRunGraph(graph_id, args, target); });
  MS_EXCEPTION_IF_NULL(result.run);

  result.simu_run = std::make_shared<RunFunc>(
    [graph_id, this](const VectorRef &args) -> VectorRef { return MsSimuRunGraph(graph_id); });
  MS_EXCEPTION_IF_NULL(result.simu_run);
  result.graph_id = graph_id;

  graph_id_map_[graph_id] = result;
  return result;
}

// compile set input output
VectorRef MsBackend::MsSimuRunGraph(const GraphId &g) {
  MS_LOG(DEBUG) << "Set graph input:" << g;
  std::vector<BaseRef> outputs;
  (void)std::transform(graph_id_map_[g].outputs.begin(), graph_id_map_[g].outputs.end(), std::back_inserter(outputs),
                       [](const AnfNodePtr &v) { return v; });
  return VectorRef(outputs);
}

namespace {
void PushInputTensor(const BaseRef &arg, std::vector<tensor::TensorPtr> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  if (utils::isa<tensor::TensorPtr>(arg)) {
    auto value = utils::cast<tensor::TensorPtr>(arg);
    inputs->push_back(value);
  } else if (utils::isa<tensor::CSRTensorPtr>(arg)) {
    auto csr = utils::cast<tensor::CSRTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(csr);
    auto csr_values = csr->GetValues();
    MS_EXCEPTION_IF_NULL(csr_values);
    inputs->push_back(csr_values);
    MS_LOG(INFO) << "For CSRTensor, push its values.";
  } else if (utils::isa<ValuePtr>(arg)) {
    auto value = utils::cast<ValuePtr>(arg);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueTuple>()) {
      auto value_tuple = value->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      auto tuple_value = value_tuple->value();
      (void)std::transform(tuple_value.begin(), tuple_value.end(), std::back_inserter(*inputs),
                           [](const ValuePtr &v) { return v->cast<tensor::TensorPtr>(); });
    } else if (value->isa<Scalar>()) {
      tensor::TensorPtr scalar_tensor = ScalarToTensor(value->cast<ScalarPtr>());
      inputs->push_back(scalar_tensor);
    } else if (value->isa<Monad>()) {
      // If value is a monad, replace it with an unused tensor.
      inputs->push_back(std::make_shared<tensor::Tensor>(int64_t(0), kBool));
    } else {
      inputs->push_back(value->cast<tensor::TensorPtr>());
    }
  } else if (utils::isa<PyObjectRef>(arg)) {
    auto value = utils::cast<PyObjectRef>(arg).object_;
    inputs->push_back(py::cast<tensor::TensorPtr>(value));
  } else if (utils::isa<VectorRefPtr>(arg)) {
    const auto &args_new = utils::cast<VectorRef>(arg);
    for (const auto &v : args_new) {
      PushInputTensor(v, inputs);
    }
  } else {
    MS_LOG(WARNING) << "Invalid input type.";
  }
}

// Insert the front_node related tensor in the input_tensor.
void PushTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                std::vector<tensor::TensorPtr> *input_tensor) {
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  if (iter == parameters.end()) {
    (void)((*input_tensor).emplace_back(nullptr));
    return;
  }
  auto position = iter - parameters.begin();
  PushInputTensor(args[position], input_tensor);
}

void UpdateOutputAbstract(const KernelGraphPtr &kernel_graph, OpRunInfo *op_run_info) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetCNodeName(kernel) == op_run_info->op_name) {
      op_run_info->abstract = kernel->abstract();
    }
  }
}

TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(output_node);
  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = AnfAlgo::GetOutputInferDataType(output_node, output_index);
  std::vector<int64_t> temp_shape;
  const auto &shape = AnfAlgo::GetOutputInferShape(output_node, output_index);
  (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
  auto tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));

  // Put device tensor into host tensor.
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  tensor->set_device_address(device_tensor);
  tensor->set_sync_status(kNeedSyncDeviceToHost);

  // MindRT is disabled in the multi graphs scenario
  // Delete tensor->data_sync() when MindRT is enabled in all scenes.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    // If execution mode is Graph Mode in MsContext, the tensor will be the input of graph which will execute in Graph
    // Mode, if the graph contain no CNode after optimization, the tensor need sync to host.
    tensor->data_sync(false);
  }

  return tensor;
}

void ClearGraphDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context, bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : graph->execution_order()) {
    auto output_address_num = AnfAlgo::GetOutputAddressNum(node);
    for (size_t i = 0; i < output_address_num; ++i) {
      if (!AnfAlgo::OutputAddrExist(node, i, false)) {
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
      if (device_address == nullptr) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(device_context);
      auto new_device_address = device_context->CreateDeviceAddress(
        nullptr, device_address->GetSize(), device_address->format(), device_address->type_id());
      MS_EXCEPTION_IF_NULL(new_device_address);
      new_device_address->set_host_shape(device_address->host_shape());
      new_device_address->set_original_ref_count(device_address->original_ref_count());
      new_device_address->ResetRefCount();
      if (is_gradient_out) {
        new_device_address->set_from_persistent_mem(true);
      }
      AnfAlgo::SetOutputAddr(new_device_address, i, node.get());
    }
  }
}

void UpdateInputDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
      AnfAlgo::SetOutputAddr(nullptr, 0, node.get());
    }
  }
}

std::vector<tensor::TensorPtr> GetRealValueNodeTensorFromGraph(
  const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &tensors_without_value_node) {
  std::vector<tensor::TensorPtr> new_input_tensors;
  if (graph->execution_order().size() != 1) {
    return new_input_tensors;
  }

  const auto &node = graph->execution_order().back();
  auto input_num = AnfAlgo::GetInputTensorNum(node);
  // No value node in graph
  if (input_num == tensors_without_value_node.size()) {
    return new_input_tensors;
  }
  MS_LOG(DEBUG) << "CNode input num:" << input_num
                << " tensors_without_value_node size:" << tensors_without_value_node.size();

  std::map<size_t, tensor::TensorPtr> value_node_pos;
  for (size_t i = 0; i < input_num; ++i) {
    auto input = AnfAlgo::GetInputNode(node, i);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<ValueNode>()) {
      auto value_node = input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<tensor::TensorPtr>();
      (void)value_node_pos.emplace(i, tensor);
    }
  }

  size_t cur_input_tensor_index = 0;
  for (size_t i = 0; i < input_num; ++i) {
    auto iter = value_node_pos.find(i);
    if (iter == value_node_pos.end()) {
      (void)new_input_tensors.emplace_back(tensors_without_value_node[cur_input_tensor_index]);
      cur_input_tensor_index++;
    } else {
      (void)new_input_tensors.emplace_back(iter->second);
    }
  }
  MS_LOG(DEBUG) << "new input tensor size:" << new_input_tensors.size();
  return new_input_tensors;
}

bool OpInBlackList(const OpRunInfo &op_run_info) {
  return kOpCacheBlackList.find(op_run_info.op_name) != kOpCacheBlackList.end();
}

int GetExecutionMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
}

bool EnablePyNativeSyncRunning() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
}

bool NeedDisableLazyBuild(bool need_erase, bool cache_hit, const OpRunInfo &op_run_info) {
  // Disable lazy build when:
  // 1. Execute Dynamic shape operator. The output shape depends on the calculation result of the operator.
  // 2. Cache hit and there are no tasks in Queue. For example Non-first iteration.
  // 3. Not in nn.Cell construct.
  // 4. Operator to process dataset.
  // 5. Graph mode.
  // 6. set PYNATIVE_SYNCHRONIZE in context.
  return need_erase || cache_hit || !op_run_info.lazy_build || OpInBlackList(op_run_info) ||
         GetExecutionMode() == kGraphMode || EnablePyNativeSyncRunning();
}
}  // namespace

VectorRef MsBackend::MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target) {
  MS_LOG(DEBUG) << "Start ms graph run:" << args.size() << ", g:" << g;
  // Run graph
  std::vector<tensor::TensorPtr> inputs;
  for (const auto &arg : args) {
    PushInputTensor(arg, &inputs);
  }

  VectorRef outputs;
  // Call ms RunGraphAsync or RunOpsInGraph (graphId, input ,output)
  const session::SessionPtr &exe_session = ((target != target_device_ && !target.empty()) ? other_sess_ : target_sess_);
  MS_EXCEPTION_IF_NULL(exe_session);
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode) {
    exe_session->RunOpsInGraph(g, inputs, &outputs);
  } else {
    exe_session->RunGraphAsync(g, inputs, &outputs);
  }

  MS_LOG(DEBUG) << "RunGraph finished:" << outputs.size();
  return outputs;
}

MsBackend::MsBackend(const std::string &name, const std::string &target, uint32_t device_id) : Backend(name) {
  convert_fn_ = std::bind(&MsBackend::MsConvert, this, std::placeholders::_1, std::placeholders::_2);
  target_sess_ = session::SessionFactory::Get().Create(target);
  if (target_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed! Please make sure target device:" << target << " is available.";
  }
  target_sess_->Init(device_id);
#ifndef ENABLE_SECURITY
  target_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
#endif
  target_device_ = target;
}

void MsBackend::CreateOtherSession(const std::string &target) {
  if (other_sess_ != nullptr && other_device_ == target) {
    return;
  }
  other_sess_ = session::SessionFactory::Get().Create(target);
  if (other_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed! Please make sure target device:" << target << " is available.";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  other_sess_->Init(device_id);
#ifndef ENABLE_SECURITY
  other_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
#endif
  other_device_ = target;
}

GraphId MsBackend::CompileGraph(NotNull<FuncGraphPtr> fg) {
  MS_EXCEPTION_IF_NULL(target_sess_);
  return target_sess_->CompileGraph(fg);
}

VectorRef MsBackend::RunGraph(GraphId graph_id, const VectorRef &args) { return MsRunGraph(graph_id, args); }

void MsBackend::ClearSessionGraphs() {
  if (target_sess_ != nullptr) {
    target_sess_->ClearGraph();
  }
}

#ifdef ENABLE_DEBUGGER
void MsBackend::SetDebugger() {
  MS_EXCEPTION_IF_NULL(target_sess_);
  target_sess_->SetDebugger();
}
#endif

MindRTBackend::MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id)
    : Backend(backend_name), device_name_(device_name) {
  root_graph_ = nullptr;
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  auto &cut_list = pynative_mode ? compile::control_ops : GetMsNonlinearOps();

  graph_partition_ = std::make_shared<GraphPartition>(cut_list, backend_name);
  graph_compiler_ = std::make_shared<GraphCompiler>();

  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  device_context->Initialize();
  device_id_ = device_context->device_context_key().device_id_;
#ifdef ENABLE_DEBUGGER
  SetDebuggerInit();
#endif
  runtime::GraphScheduler::GetInstance().Initialize();
}

const ActorInfo &MindRTBackend::CompileGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Status record: start compile function graph: " << func_graph->ToString();
  PROF_START(compile_func_graph);
  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  root_graph_ = root_graph.get();
  // Register a summary callback function, which is called in the final stages of summary.
  graph_compiler_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  ms_execution_mode_ = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  real_execution_mode_ = ms_execution_mode_;
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  auto is_parallel = (parallel_mode == parallel::SEMI_AUTO_PARALLEL || parallel_mode == parallel::AUTO_PARALLEL);

  // Run in GRAPH_MODE if the func_graph is ms_function or the func_graph contain multi-subgraph.
  if (ms_execution_mode_ == kPynativeMode &&
      (!func_graph->is_bprop() || func_graph->manager()->func_graphs().size() > 1) && !is_parallel) {
    real_execution_mode_ = kGraphMode;
    context_ptr->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
    pipeline::SetRunMode(func_graph, this);
    MS_LOG(INFO) << "PyNative graph Compile and Run in GRAPH_MODE";
  }

  // Compile root graph.
  graph_id_to_device_context_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  control_nodes_.clear();
  auto subgraph_need_compile = CompileGraph(root_graph);
  // Compile sub graphs.
  if (subgraph_need_compile) {
    MS_EXCEPTION_IF_NULL(root_graph->manager());
    FuncGraphSet sub_graphs = root_graph->manager()->func_graphs();
    for (auto sub_graph : sub_graphs) {
      if (sub_graph != func_graph && sub_graph != nullptr) {
        (void)CompileGraph(sub_graph);
      }
    }
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  if (real_execution_mode_ == kGraphMode && graph_compiler_info->graphs_.size() != 0) {
    // Transform graph to actor DAG, and schedule the actor DAG.
    const auto &actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
    runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  }
  const ActorInfo &actor_info = graph_compiler_info->name_;
  (void)actor_to_graph_compiler_info_.emplace(graph_compiler_info->name_, std::move(graph_compiler_info));
  PROF_END(compile_func_graph);

  if (ms_execution_mode_ != real_execution_mode_) {
    context_ptr->set_param<int>(MS_CTX_EXECUTION_MODE, ms_execution_mode_);
  }

  MS_LOG(INFO) << "Status record: end compile function graph: " << func_graph->ToString()
               << ", produce actor: " << actor_info;
  return actor_info;
}

bool MindRTBackend::CompileGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_partition_);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  bool contain_multi_target = false;
  // Split graph to segments.
  const auto &segments = graph_partition_->Partition(func_graph, &contain_multi_target);
  MS_LOG(INFO) << "Compile graph: " << func_graph->ToString() << ", Split segments size:" << segments.size();
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &new_segments = device_context->PartitionGraph(func_graph, segments);

  // Compile the whole function graph if not split graph.
  if (new_segments.size() == 0) {
    auto graph_id = graph_compiler_->CompileGraph(func_graph, device_context);
    graph_id_to_device_context_[graph_id] = device_context;
    return false;
  }

  // Foreach the segments to compile graph.
  for (const auto &segment : new_segments) {
    CompileGraph(segment);
  }
  return true;
}

void MindRTBackend::CompileGraph(const GraphSegmentPtr &segment) {
  MS_EXCEPTION_IF_NULL(segment);
  // Compile the normal nodes, which doesn't contain the cut node.
  if (segment->nodes_.size() == 0) {
    MS_LOG(EXCEPTION) << "The segments size is 0.";
  }
  if (!segment->is_cut_) {
    MS_EXCEPTION_IF_NULL(segment->nodes_[0]);
    MS_LOG(INFO) << "Compile normal segment, the first node: " << segment->nodes_[0]->DebugString();

    // Get the device context.
    const auto &cur_device_name = GetCNodeTarget(segment->nodes_[0]);
    const auto &device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({cur_device_name, device_id_});
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->Initialize();

    // Transform nodes to inputs and outputs.
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(segment->nodes_);

    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    // Compile graph.
    auto graph_id =
      graph_compiler_->CompileGraph(segment, outputs, device_context, real_execution_mode_ == kPynativeMode);

    graph_id_to_device_context_[graph_id] = device_context;

    const auto &func_graph = segment->nodes_[0]->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph_to_kernel_graph_ids_.find(func_graph) == func_graph_to_kernel_graph_ids_.end()) {
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>{graph_id});
    } else {
      (void)func_graph_to_kernel_graph_ids_[func_graph].back().emplace_back(graph_id);
    }
  } else {
    // Compile the cut node.
    auto cut_node = segment->nodes_[0];
    MS_EXCEPTION_IF_NULL(cut_node);
    MS_LOG(INFO) << "Compile cut segment, the cut node: " << cut_node->DebugString();
    control_nodes_.push_back(cut_node);
    if (AnfAlgo::IsCallNode(cut_node) || AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitchLayer)) {
      const auto &func_graph = cut_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>());
    }
  }
}

namespace {
ValuePtr GetControlOpInputFromMakeTuple(const std::shared_ptr<GraphCompiler> &graph_compiler,
                                        const AnfNodePtr &front_cnode, const CNodePtr &backend_cnode,
                                        const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                                        const std::map<AnfNodePtr, size_t> &parameter_index,
                                        const std::vector<tensor::TensorPtr> &graph_inputs,
                                        InputTensorInfo *input_tensor_info, size_t *input_index) {
  MS_EXCEPTION_IF_NULL(graph_compiler);
  MS_EXCEPTION_IF_NULL(front_cnode);
  MS_EXCEPTION_IF_NULL(input_index);
  MS_LOG(DEBUG) << "The input node of hook op: " << front_cnode->DebugString() << " is a make tuple node.";
  auto make_tuple = front_cnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple);
  const auto output_size = make_tuple->size() - 1;
  std::vector<ValuePtr> output_values;
  for (size_t idx = 0; idx < output_size; ++idx) {
    TensorPtr tensor = graph_compiler->GetSingleOpInputTensorByIndex(backend_cnode, op_output_map, parameter_index,
                                                                     graph_inputs, input_tensor_info, *input_index);
    MS_EXCEPTION_IF_NULL(tensor);
    output_values.emplace_back(tensor);
    ++(*input_index);
  }
  return std::make_shared<ValueTuple>(output_values);
}

void GetControlOpInput(const std::shared_ptr<GraphCompiler> &graph_compiler, const CNodePtr &front_cnode,
                       const CNodePtr &backend_cnode, const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                       const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                       VectorRef *args) {
  MS_EXCEPTION_IF_NULL(front_cnode);
  MS_EXCEPTION_IF_NULL(backend_cnode);
  MS_EXCEPTION_IF_NULL(graph_compiler);
  MS_EXCEPTION_IF_NULL(args);
  size_t input_index = 0;
  auto inputs = front_cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    const auto &input_node = inputs[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
      // Hook multi-input or multi-output.
      args->emplace_back(GetControlOpInputFromMakeTuple(graph_compiler, input_node, backend_cnode, op_output_map,
                                                        parameter_index, graph_inputs, input_tensor_info,
                                                        &input_index));
      continue;
    }
    // Hook single-input or single-output.
    auto real_input = AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(real_input);
    if (!real_input->isa<ValueNode>()) {
      auto tensor = graph_compiler->GetSingleOpInputTensorByIndex(backend_cnode, op_output_map, parameter_index,
                                                                  graph_inputs, input_tensor_info, input_index);
      MS_EXCEPTION_IF_NULL(tensor);
      args->emplace_back(tensor);
      ++input_index;
    } else {
      const auto &value_node = real_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      const auto &value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      args->emplace_back(value);
      if (value->isa<ValueSequence>()) {
        const auto &value_sequeue = value->cast<ValueSequencePtr>();
        MS_EXCEPTION_IF_NULL(value_sequeue);
        input_index += value_sequeue->size();
      } else {
        ++input_index;
      }
    }
  }
}

void ConvertPyObjectToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    double input_value = py::cast<py::float_>(input_object);
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
  } else if (py::isinstance<py::int_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<int64_t>(input_object), kInt64);
  } else if (py::isinstance<py::list>(input_object)) {
    auto list_inputs = py::cast<py::list>(input_object);
    for (size_t i = 0; i < list_inputs.size(); ++i) {
      ConvertPyObjectToTensor(list_inputs[i], tensors);
    }
    return;
  } else if (py::isinstance<py::tuple>(input_object)) {
    auto tuple_inputs = py::cast<py::tuple>(input_object);
    for (size_t i = 0; i < tuple_inputs.size(); ++i) {
      ConvertPyObjectToTensor(tuple_inputs[i], tensors);
    }
    return;
  } else {
    MS_EXCEPTION(TypeError) << "Unreasonable data type: " << input_object.get_type() << ".";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)tensors->emplace_back(tensor_ptr);
}

void RunControlOperator(const std::shared_ptr<GraphCompiler> &graph_compiler, const KernelGraphPtr &graph,
                        const CNodePtr &kernel, const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                        const std::map<AnfNodePtr, size_t> &parameter_index,
                        const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                        VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_outputs);
  AnfNodePtr front_node = graph->GetFrontAnfByBackendAnf(kernel);
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The front node of bprop_cut is not CNode";
  }
  CNodePtr cnode = front_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const std::vector<AnfNodePtr> &node_inputs = cnode->inputs();
  if (node_inputs.empty()) {
    MS_LOG(EXCEPTION) << "The inputs of node[" << cnode->fullname_with_scope() << "] is empty";
  }

  const AnfNodePtr &fn = node_inputs.at(0);
  if (!IsValueNode<Primitive>(fn)) {
    MS_LOG(EXCEPTION) << "The input[0] of kernel[" << kernel->fullname_with_scope()
                      << "] is not a ValueNode of Primitive";
  }

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(fn);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == kBpropCutOpName) {
    VectorRef args;
    GetControlOpInput(graph_compiler, cnode, kernel, op_output_map, parameter_index, graph_inputs, input_tensor_info,
                      &args);
    auto py_prim = prim->cast<PrimitivePyPtr>();
    MS_EXCEPTION_IF_NULL(py_prim);
    BaseRef out = py_prim->RunHookFunction(args);
    // Convert pyobject output to tensor.
    if (utils::isa<PyObjectRef>(out)) {
      PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
      auto out_py_tuple = py_ref.object_;
      std::vector<tensor::TensorPtr> output_tensors;
      ConvertPyObjectToTensor(out_py_tuple, &output_tensors);
      (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(op_outputs->elements_),
                           [](tensor::TensorPtr &tensor) { return std::move(tensor); });
    }
  }
}

void TensorValueToVector(const ValuePtr &value, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(outputs);
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        outputs->emplace_back(tensor);
      } else if (element->isa<ValueTuple>()) {
        TensorValueToVector(element, outputs);
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    outputs->emplace_back(tensor);
  }
}

bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &graph_output, const VectorRef &args, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(graph_output);
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    VectorRef output_tmp;
    ValuePtr value = GetValueNode(graph_output);
    TensorValueToVector(value, &output_tmp);
    if (output_tmp.size() == 1) {
      *outputs = std::move(output_tmp);
    } else if (output_tmp.size() > 1) {
      outputs->emplace_back(output_tmp);
    } else {
      MS_LOG(EXCEPTION) << "Output is empty!";
    }
    return true;
  }

  if (graph_output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    // Find the right parameter as ret_val.
    auto func_graph = graph_output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if (args.size() != params.size()) {
      MS_LOG(EXCEPTION) << "Input size " << args.size() << " not equal to graph input size " << params.size();
    }

    auto it = std::find(params.begin(), params.end(), graph_output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter, it should be found in graph parameters";
    }
    size_t index = it - params.cbegin();
    if (index >= args.size()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size();
    }

    outputs->emplace_back(args[index]);
    return true;
  }
  return false;
}
}  // namespace

void FlatValueTupleValue(const ValuePtrList &value, ValuePtrList *flatted_value) {
  for (size_t i = 0; i < value.size(); ++i) {
    auto value_element = value[i];
    MS_EXCEPTION_IF_NULL(value_element);
    if (utils::isa<tensor::TensorPtr>(value_element)) {
      (void)flatted_value->emplace_back(value_element);
    } else if (utils::isa<ValueTuplePtr>(value_element)) {
      auto value_tuple_element = value_element->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple_element);
      FlatValueTupleValue(value_tuple_element->value(), flatted_value);
    } else {
      MS_LOG(EXCEPTION) << "The value input to FlatValueTupleValue should only contains Tensor and ValueTuple.";
    }
  }
}

void PushTupleTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                     size_t index, std::vector<tensor::TensorPtr> *input_tensor) {
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  const size_t position = iter - parameters.begin();
  // If the parameter is not found in the parameters of the root graph, it means that it is the input of the subgraph,
  // and there is no need to input a tensor.
  if (position >= args.size()) {
    MS_LOG(INFO) << "Position out of args range, position value is " << position << " and args size is " << args.size()
                 << ".";
    (void)input_tensor->emplace_back(nullptr);
    return;
  }
  auto value_tuple = utils::cast<ValueTuplePtr>(args[position]);
  MS_EXCEPTION_IF_NULL(value_tuple);
  auto value_tuple_value = value_tuple->value();
  ValuePtrList flatted_value_tuple_value;
  FlatValueTupleValue(value_tuple_value, &flatted_value_tuple_value);
  if (index >= flatted_value_tuple_value.size()) {
    MS_LOG(EXCEPTION) << "Index out of flatted_value_tuple_value range, index value is " << index
                      << " and flatted_value_tuple_value size is " << flatted_value_tuple_value.size() << ".";
  }
  auto input = flatted_value_tuple_value[index];
  MS_EXCEPTION_IF_NULL(input);
  auto tensor_input = input->cast<tensor::TensorPtr>();
  input_tensor->push_back(tensor_input);
}

void MindRTBackend::RunGraphBySingleOp(const std::vector<KernelGraphPtr> &graphs,
                                       const std::vector<std::vector<tensor::TensorPtr>> &inputs, VectorRef *outputs) {
  SyncLazyTasks();
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto &op_lazy_builder = runtime::OpLazyBuilder::GetInstance();
  op_lazy_builder.Register([this]() { LazyExecuteTaskCallback(); });
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    MS_EXCEPTION_IF_NULL(graph);
    std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;
    std::map<AnfNodePtr, size_t> parameter_index;
    GraphOutputInfo graph_output_info;
    graph_output_info.graph_outputs = outputs;
    graph_compiler_->GetParamAndOutputIndex(graph, inputs[graph_index], outputs, &parameter_index,
                                            &graph_output_info.output_indexes);

    std::map<KernelWithIndex, size_t> cnode_ref_count;
    auto iter = cnode_ref_counts_.find(graph->graph_id());
    if (iter == cnode_ref_counts_.end()) {
      graph_compiler_->CalculateRefCount(graph, &cnode_ref_count);
      (void)cnode_ref_counts_.emplace(graph->graph_id(), cnode_ref_count);
    } else {
      cnode_ref_count = iter->second;
    }
    graph_compiler_->CalculateForwardOpOutputCount(graph, inputs[graph_index], &forward_op_output_tensor_id_);

    for (const auto &kernel : graph->execution_order()) {
      InputTensorInfo input_tensor_info;
      VectorRef op_outputs;

      if (!AnfAlgo::IsControlOpExecInBackend(kernel)) {
        OpRunInfo op_run_info;
        GraphInfo graph_info;
        graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index],
                                                 &input_tensor_info);
        graph_compiler_->GetSingleOpRunInfoAndGraphInfo(kernel, input_tensor_info, &op_run_info, &graph_info,
                                                        &graph_output_info);

        RunOp(&op_run_info, &op_outputs);
      } else {
        SyncLazyTasks();
        RunControlOperator(graph_compiler_, graph, kernel, op_output_map, parameter_index, inputs[graph_index],
                           &input_tensor_info, &op_outputs);
        // Execute remaining lazy tasks before PyNative hook exit.
        SyncLazyTasks();
      }

      graph_compiler_->UpdateRefCount(input_tensor_info.input_kernel, &cnode_ref_count, &op_output_map);

      graph_output_info.graph_output_tensors.clear();
      graph_compiler_->RecoverGraphOutput(kernel, op_outputs, cnode_ref_count, &op_output_map, &graph_output_info);

      // Save grad node to Bucket
      if (graph->is_bprop() && (!AnfAlgo::IsControlOpExecInBackend(kernel)) && !kernel->is_parallel()) {
        graph_compiler_->AddGradAddrToBucket(graph->graph_id(), graph_output_info.graph_output_tensors);
      }
    }
    SyncLazyTasks();
    // Clear bucket resources every step
    if (graph->is_bprop()) {
      graph_compiler_->ClearAllBucket(graph->graph_id());
    }
  }
}

void MindRTBackend::RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(root_graph_);
  if (IsGraphOutputValueNodeOrParameter(root_graph_->output(), args, outputs)) {
    return;
  }

  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return;
  }

  // Open abstract_lock for dynamic_shape
  AnfUtils::OpenAbstractLock();

  MS_LOG(INFO) << "Status record: start run actor: " << actor_info;
  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info.";
  }
  MS_EXCEPTION_IF_NULL(graph_iter->second);
  const auto &graph_compiler_info = *(graph_iter->second);
  const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;

  SyncLazyTasks();

  // Transform args to input tensors.
  // Input tensors of the graph.
  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    std::vector<tensor::TensorPtr> input_tensor;
    MS_EXCEPTION_IF_NULL(kernel_graph);
    for (const auto &input_node : kernel_graph->input_nodes()) {
      auto element_pair = kernel_graph->GetElementInTupleBackendFrontIndexMap(input_node);
      if (element_pair.first) {
        PushTupleTensor(args, origin_parameters, element_pair.first, element_pair.second, &input_tensor);
      } else {
        const auto &front_node = kernel_graph->GetFrontAnfByBackendAnf(input_node);
        PushTensor(args, origin_parameters, front_node, &input_tensor);
      }
    }
    (void)input_tensors.emplace_back(input_tensor);
  }

  // Input tensors of the control node.
  std::vector<tensor::TensorPtr> input_tensor;
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  // Get inputs of control node which come from the host actor.
  const auto &control_node_parameters = graph_compiler_info.control_node_parser_->control_node_parameters();
  for (const auto &parameter : control_node_parameters) {
    PushTensor(args, origin_parameters, parameter, &input_tensor);
  }
  (void)input_tensors.emplace_back(input_tensor);

  // Run in the pynative mode.
  MS_EXCEPTION_IF_NULL(outputs);
  // There will be more than one kernel graph in heterogeneous scenario in a ms function of PyNative Mode.
  if (real_execution_mode_ == kPynativeMode) {
    RunGraphBySingleOp(graph_compiler_info.graphs_, input_tensors, outputs);
    MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
    return;
  }

  // Run actor DAG.
  mindspore::ScopedLongRunning long_running;
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  runtime::GraphScheduler::GetInstance().Run(actor_set, graph_compiler_info.device_contexts_, input_tensors);

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  // Update device address for output node of graph.
  // Summary processing will use the output device address, so must be after the summary processing.
  actor_set->output_actor_->UpdateOutputDeviceAddress();

  // Fetch outputs.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  auto &output_tensors = actor_set->output_actor_->outputs();
  if (output_tensors.size() > 0) {
    size_t output_position = 0;
    ConstructOutputs(root_graph_->output(), output_tensors, &output_position, outputs);
  }
  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
}

BaseRef MindRTBackend::ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                                 const std::vector<tensor::TensorPtr> &output_tensors,
                                                 size_t *output_position) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(output_position);

  size_t outputs_num = AnfAlgo::GetOutputNumByAbstract(abstract);
  if (*output_position + outputs_num > output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position << " need:" << outputs_num
                      << " total:" << output_tensors.size();
  }
  VectorRef outputs;

  if (abstract->isa<abstract::AbstractCSRTensor>()) {
    auto csr_tensor_abstract = abstract->cast<abstract::AbstractCSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor_abstract);
    outputs.emplace_back(ConstructOutputByAbstract(csr_tensor_abstract->indptr(), output_tensors, output_position));
    outputs.emplace_back(ConstructOutputByAbstract(csr_tensor_abstract->indices(), output_tensors, output_position));
    outputs.emplace_back(ConstructOutputByAbstract(csr_tensor_abstract->values(), output_tensors, output_position));
    outputs.emplace_back(
      ConstructOutputByAbstract(csr_tensor_abstract->dense_shape(), output_tensors, output_position));
    return outputs;
  }

  if (abstract->isa<abstract::AbstractCOOTensor>()) {
    auto coo_tensor_abstract = abstract->cast<abstract::AbstractCOOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor_abstract);
    outputs.emplace_back(ConstructOutputByAbstract(coo_tensor_abstract->indices(), output_tensors, output_position));
    outputs.emplace_back(ConstructOutputByAbstract(coo_tensor_abstract->values(), output_tensors, output_position));
    outputs.emplace_back(
      ConstructOutputByAbstract(coo_tensor_abstract->dense_shape(), output_tensors, output_position));
    return outputs;
  }

  if (!abstract->isa<abstract::AbstractTuple>()) {
    (*output_position)++;
    return output_tensors[(*output_position) - 1];
  }

  auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  const auto &sub_abstracts = tuple_abstract->elements();
  for (const auto &sub_abstract : sub_abstracts) {
    MS_EXCEPTION_IF_NULL(sub_abstract);
    outputs.emplace_back(ConstructOutputByAbstract(sub_abstract, output_tensors, output_position));
  }
  return outputs;
}

void MindRTBackend::ConstructOutputs(const AnfNodePtr &output_node,
                                     const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                     VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_position);
  const PrimitiveSet expand_prims{
    prim::kPrimMakeTuple,
    prim::kPrimMakeCSRTensor,
    prim::kPrimMakeCOOTensor,
    prim::kPrimMakeRowTensor,
  };
  // The MakeTuple/MakeSaprse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(output_node, expand_prims)) {
    auto make_tuple = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    VectorRef make_tuple_output;
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      ConstructOutputs(make_tuple->input(i), output_tensors, output_position, &make_tuple_output);
    }
    outputs->emplace_back(std::move(make_tuple_output));
    return;
  }

  // The depend node need get the real node.
  if (AnfAlgo::CheckPrimitiveType(output_node, prim::kPrimDepend)) {
    auto depend_node = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
    ConstructOutputs(depend_node->input(kRealInputIndexInDepend), output_tensors, output_position, outputs);
    return;
  }

  auto outputs_num = AnfAlgo::GetOutputTensorNum(output_node);
  // The value node uses the value to be output, to avoid the host memory of value free due to value node destruction.
  if (output_node->isa<ValueNode>()) {
    auto value = output_node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueTuple>()) {
      outputs->emplace_back(value);
      (*output_position) += CountValueNum(value->cast<ValueTuplePtr>());
    } else if (outputs_num != 0) {
      outputs->emplace_back(value);
      (*output_position) += outputs_num;
    }
    // The empty value node return the empty VectorRef.
    return;
  }

  if (AnfAlgo::IsCallNode(output_node)) {
    auto abstract = output_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    outputs->emplace_back(ConstructOutputByAbstract(abstract, output_tensors, output_position));
    return;
  }

  auto &output_abstract = output_node->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  // Wrap output to VectorRef if the output is tuple.
  if (output_abstract->isa<abstract::AbstractTuple>()) {
    VectorRef output_tuple;
    for (size_t i = 0; i < outputs_num; ++i) {
      if (*output_position >= output_tensors.size()) {
        MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position;
      }
      output_tuple.emplace_back(std::move(output_tensors[*output_position]));
      ++(*output_position);
    }
    outputs->emplace_back(std::move(output_tuple));
  } else {
    for (size_t i = 0; i < outputs_num; ++i) {
      if (*output_position >= output_tensors.size()) {
        MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position;
      }
      outputs->emplace_back(std::move(output_tensors[*output_position]));
      ++(*output_position);
    }
  }
}

#ifdef ENABLE_DEBUGGER
void MindRTBackend::SetDebuggerInit() {
  auto debugger_ = Debugger::GetInstance();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  debugger_->Init(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
}
#endif

void MindRTBackend::SyncLazyTasks() const { runtime::OpLazyBuilder::GetInstance().ExecuteRemainingTasks(); }

void MindRTBackend::ClearOpBuilderResource() const { runtime::OpLazyBuilder::GetInstance().Reset(); }

void MindRTBackend::SyncStream() {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  (void)device_context->SyncStream();
}

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::string name = "kernel_graph";
  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    (void)graphs.emplace_back(graph_compiler_->Fetch(graph_id_to_context.first));
    (void)device_contexts.emplace_back(graph_id_to_context.second);
    (void)name.append("_").append(std::to_string(graph_id_to_context.first));
  }

  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_ids : func_graph_to_kernel_graph_ids_) {
    const auto &func_graph = func_graph_to_kernel_graph_ids.first;
    for (const auto &sub_kernel_graphs_ids : func_graph_to_kernel_graph_ids.second) {
      std::vector<KernelGraphPtr> kernel_graphs;
      for (const auto &graph_id : sub_kernel_graphs_ids) {
        const auto &kernel_graph = graph_compiler_->Fetch(graph_id);
        MS_EXCEPTION_IF_NULL(kernel_graph);
        (void)kernel_graphs.emplace_back(kernel_graph);
      }
      (void)func_graph_to_kernel_graphs[func_graph].emplace_back(kernel_graphs);
    }
  }

  auto parser = std::make_shared<ControlNodeParser>();
  parser->Parse(control_nodes_, graphs, device_contexts, root_graph, func_graph_to_kernel_graphs);

  runtime::KernelMapPosition outputs_order;
  const auto &root_output =
    AnfAlgo::VisitKernelWithReturnType(root_graph->output(), 0, false, {prim::kPrimTupleGetItem}).first;
  size_t position = 0;
  auto outputs = AnfAlgo::GetAllOutputWithIndex(root_output);
  size_t outputs_num = outputs.size();
  for (const auto &output : outputs) {
    if (outputs_order.count(output) == 0) {
      outputs_order[output] = {position++};
    } else {
      (void)outputs_order[output].emplace_back(position++);
    }
  }

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes_,
                                             root_graph->parameters(), parser, outputs_order, outputs_num, name, false,
                                             runtime::GraphExecutionStrategy::kPipeline);
}

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(
  const ActorInfo &actor_info, const std::vector<int64_t> *tensors_mask,
  const std::vector<tensor::TensorPtr> *input_tensors, bool need_erase) {
  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  runtime::KernelMapPosition outputs_order;
  size_t position = 0;
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  for (const auto &graph_info_to_context : graph_info_to_device_context_) {
    const auto &graph = graph_compiler_->Fetch(graph_info_to_context.first);
    MS_EXCEPTION_IF_NULL(graph);
    (void)graphs.emplace_back(graph);
    (void)device_contexts.emplace_back(graph_info_to_context.second);

    auto outputs = AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output : outputs) {
      if (outputs_order.count(output) == 0) {
        outputs_order[output] = {position++};
      } else {
        (void)outputs_order[output].emplace_back(position++);
      }
    }
  }

  std::vector<std::vector<int64_t> *> tensors_mask_list(1, const_cast<std::vector<int64_t> *>(tensors_mask));
  std::vector<std::vector<TensorPtr> *> input_tensors_list(1,
                                                           const_cast<std::vector<tensor::TensorPtr> *>(input_tensors));
  auto parser = std::make_shared<ControlNodeParser>();
  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask_list, input_tensors_list,
                                             std::vector<AnfNodePtr>(), std::vector<AnfNodePtr>(), parser,
                                             outputs_order, 0, actor_info, need_erase,
                                             runtime::GraphExecutionStrategy::kStep);
}

void MindRTBackend::EraseSingleOpCache(const ActorInfo &actor_info, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph_info_to_device_context_.empty()) {
    MS_LOG(EXCEPTION) << "The map graph_info_to_device_context_ is empty.";
  }
  const auto &graph_info = graph_info_to_device_context_.begin()->first;
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->EraseSingleOpCache(graph_info, graph->graph_id());
  actor_to_graph_compiler_info_.erase(actor_info);
}

void MindRTBackend::RunSingleOpGraph(const KernelGraphPtr &graph, const OpRunInfo &op_run_info,
                                     const GraphCompilerInfo *graph_compiler_info) {
  // Erase value node tensor.
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  const auto &input_tensors = op_run_info.input_tensors;
  const auto &tensors_mask = op_run_info.tensor_mask;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t index = 0; index < tensors_mask.size(); ++index) {
    if (tensors_mask.at(index) != kValueNodeTensorMask) {
      (void)tensors_without_value_node.emplace_back(input_tensors.at(index));
    }
  }

  std::vector<tensor::TensorPtr> new_input_tensors = GetRealValueNodeTensorFromGraph(graph, tensors_without_value_node);

  for (auto &tensor : tensors_without_value_node) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->NeedWaitDevice()) {
      tensor->WaitDevice();
    }
  }

  // Run actor DAG.
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_compiler_info->name_);
  MS_EXCEPTION_IF_NULL(actor_set);
  runtime::GraphScheduler::GetInstance().Run(actor_set, {}, {tensors_without_value_node},
                                             new_input_tensors.empty() ? input_tensors : new_input_tensors,
                                             runtime::GraphExecutionStrategy::kStep);

  // Release the kernel resource.
  const auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (kOpCacheBlackList.find(AnfAlgo::GetCNodeName(kernel)) != kOpCacheBlackList.end()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
      if (kernel_mod) {
        kernel_mod->ReleaseResource();
      }
    }
  }

  // Update forward op output ref counts, release it
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    graph_compiler_->UpdateForwardOpOutputRefCount(input_tensors, &forward_op_output_tensor_id_);
  }
}

void MindRTBackend::CompileSingleOpGraphs(const std::vector<std::shared_ptr<runtime::OpTask>> &build_tasks) {
  if (build_tasks.empty()) {
    return;
  }
  std::vector<KernelGraphPtr> graphs;
  std::vector<GraphCompilerInfo *> graph_compiler_infos;
  for (const auto &task : build_tasks) {
    MS_EXCEPTION_IF_NULL(task);
    const auto &context = task->context();
    MS_EXCEPTION_IF_NULL(context);
    graphs.push_back(context->graph());
    graph_compiler_infos.push_back(context->graph_compiler_info());
  }
  MS_EXCEPTION_IF_NULL(build_tasks[0]);
  auto &task_context = build_tasks[0]->context();
  MS_EXCEPTION_IF_NULL(task_context);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, task_context->is_pynative_infer());

  auto device_context = task_context->device_context();
  graph_compiler_->BuildSingleOpGraphs(graphs, device_context);

  for (const auto &graph_compiler_info : graph_compiler_infos) {
    MS_EXCEPTION_IF_NULL(graph_compiler_info);
    auto actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
    graph_compiler_info->input_tensors_.clear();
    runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  }
}

void MindRTBackend::LazyExecuteTaskCallback() {
  auto &op_lazy_builder = runtime::OpLazyBuilder::GetInstance();
  if (op_lazy_builder.QueueEmpty()) {
    return;
  }

  try {
    MS_LOG(DEBUG) << "Start";
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);

    CompileSingleOpGraphs(op_lazy_builder.GetOpBuildTasks());
    op_lazy_builder.ClearOpBuildTasks();

    // Run op one by one
    auto &op_run_tasks = op_lazy_builder.GetOpRunTasks();
    while (!op_run_tasks.empty()) {
      auto &op_run_task = op_run_tasks.front();
      const auto &context = op_run_task->context();
      ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());
      RunSingleOpGraph(context->graph(), context->op_run_info(), context->graph_compiler_info());
      ClearGraphDeviceAddress(context->graph(), context->device_context(), context->op_run_info().is_gradient_out);

      UpdateInputDeviceAddress(context->graph());

      op_lazy_builder.PopOpRunTask();
    }

    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
    MS_LOG(DEBUG) << "End";
  } catch (const py::type_error &ex) {
    op_lazy_builder.Reset();
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    op_lazy_builder.Reset();
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    op_lazy_builder.Reset();
    throw py::index_error(ex);
  } catch (const py::name_error &ex) {
    op_lazy_builder.Reset();
    throw py::name_error(ex);
  } catch (const std::exception &ex) {
    op_lazy_builder.Reset();
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    op_lazy_builder.Reset();
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when execute task in queue. Exception name: " << exName;
  }
}

void MindRTBackend::RunOpInternal(bool single_op_cache_hit, GraphCompilerInfo *graph_compiler_info,
                                  OpRunInfo *op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  // Fetch outputs.
  const auto &graph = graph_compiler_info->graphs_.front();
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  const auto &output_nodes = graph_compiler_->GetGraphOutputNodes(graph->graph_id());
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = graph_compiler_info->device_contexts_.front();
  auto &op_lazy_builder = runtime::OpLazyBuilder::GetInstance();

  bool lazy_build_disabled = NeedDisableLazyBuild(graph_compiler_info->need_erase_,
                                                  (single_op_cache_hit && op_lazy_builder.QueueEmpty()), *op_run_info);
  if (lazy_build_disabled) {
    if (!op_lazy_builder.QueueEmpty()) {
      op_lazy_builder.ExecuteRemainingTasks();
    }
    if (!single_op_cache_hit) {
      CompileSingleOpGraph(graph, device_context, graph_compiler_info);
    }
    RunSingleOpGraph(graph, *op_run_info, graph_compiler_info);
    UpdateOutput(output_nodes, outputs);
    ClearGraphDeviceAddress(graph, device_context, op_run_info->is_gradient_out);
    UpdateInputDeviceAddress(graph);
    if (op_run_info->is_dynamic_shape) {
      UpdateOutputAbstract(graph, op_run_info);
    }
    if (graph_compiler_info->need_erase_) {
      EraseSingleOpCache(graph_compiler_info->name_, graph);
    }
  } else {
    UpdateOutput(output_nodes, outputs);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
    auto run_op_context =
      std::make_shared<runtime::OpLazyBuilderContext>(graph_compiler_info, graph, output_nodes, *op_run_info,
                                                      graph_compiler_info->device_contexts_.front(), infer_flag);
    if (!single_op_cache_hit) {
      op_lazy_builder.PushOpBuildTask(std::make_shared<runtime::OpBuildTask>(run_op_context));
    }
    op_lazy_builder.PushOpRunTask(std::make_shared<runtime::OpRunTask>(run_op_context));
    // Callbacks need to be re-registered in heterogeneous scenarios.
    op_lazy_builder.Register([this]() { LazyExecuteTaskCallback(); });
    if (op_lazy_builder.QueueFull()) {
      op_lazy_builder.ExecuteRemainingTasks();
    }
  }
}

void MindRTBackend::RunOp(OpRunInfo *op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  // Get the device context.
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  bool single_op_cache_hit = true;
  auto graph_id = graph_compiler_->CompileGraph(*op_run_info, &single_op_cache_hit, device_context);
  std::string actor_info = std::to_string(graph_id) + "_" + op_run_info->op_name;
  GraphCompilerInfo *graph_compiler_info_ptr;
  if (single_op_cache_hit) {
    auto iter = actor_to_graph_compiler_info_.find(actor_info);
    if (iter == actor_to_graph_compiler_info_.end()) {
      MS_LOG(EXCEPTION) << "Can not find graph compiler info for actor set: " << actor_info;
    }
    graph_compiler_info_ptr = iter->second.get();
  } else {
    graph_info_to_device_context_.clear();
    graph_info_to_device_context_[op_run_info->graph_info] = device_context;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool enable_cache = context_ptr->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
    auto graph_compiler_info =
      ConstructGraphCompilerInfo(actor_info, &op_run_info->tensor_mask, &op_run_info->input_tensors, !enable_cache);
    graph_compiler_info_ptr = graph_compiler_info.get();

    auto ret = actor_to_graph_compiler_info_.try_emplace(actor_info, std::move(graph_compiler_info));
    if (!ret.second) {
      MS_LOG(WARNING) << "ActorInfo:" << actor_info << " already exist in the map.";
    }
  }

  RunOpInternal(single_op_cache_hit, graph_compiler_info_ptr, op_run_info, outputs);
}

void MindRTBackend::CompileSingleOpGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                         GraphCompilerInfo *graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  graph_compiler_->BuildSingleOpGraphs({graph}, device_context);
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  auto actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
  graph_compiler_info->input_tensors_.clear();
  // Actor::Init() is called in Schedule.
  // Workspace need to be initialized in Actor::Init().
  // So `Schedule` need to execute after `CreateKernelWorkspaceDeviceAddress`.
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);
}

void MindRTBackend::UpdateOutput(const std::vector<session::KernelWithIndex> &output_nodes, VectorRef *const outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto output_tensor = CreateOutputTensor(item_with_index.first, item_with_index.second);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_lazy_callback([]() { runtime::OpLazyBuilder::GetInstance().ExecuteRemainingTasks(); });
    outputs->emplace_back(output_tensor);
  }
}
}  // namespace compile
}  // namespace mindspore
