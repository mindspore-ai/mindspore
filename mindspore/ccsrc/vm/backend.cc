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
#include "vm/backend.h"

#include <algorithm>
#include <vector>
#include <map>

#include "vm/transform.h"
#include "backend/session/session_factory.h"
#include "pipeline/pynative/pynative_execute.h"
#include "ir/anf.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/callbacks.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/framework/graph_compiler.h"
#include "utils/scoped_long_running.h"
#ifdef ENABLE_GE
#include "utils/callbacks_ge.h"
#endif

namespace mindspore {
namespace compile {
bool Backend::GetCond(const BaseRef &c, bool *const value) { return BaseRefToBool(c, value); }
bool Backend::GetIndex(const BaseRef &c, int64_t *const value) { return BaseRefToInt(utils::cast<ValuePtr>(c), value); }

Backend::Backend(const std::string &name) : name_(name) {
  MS_LOG(DEBUG) << "select backend:" << name;
  convert_fn_ = MsVmConvert;
  is_multi_graph_sink_ = false;
}

LinConvertResult MsBackend::MsConvert(const GraphSegmentPtr &segment, const std::string &target) {
  MS_LOG(DEBUG) << "MsConvert";
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  auto cached = g_ConvertCache.find(segment);
  if (cached != g_ConvertCache.end()) {
    return cached->second;
  }
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
    auto pre_graph = target_sess_->GetGraph(pre_segment->graph_id_);
    if (pre_graph == nullptr) {
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
      other_sess_->BuildGraph(graph_id);
    } else if (!is_multi_graph_sink_) {
      target_sess_->BuildGraph(graph_id);
    }
  }
  result.run = std::make_shared<RunFunc>(
    [graph_id, target, this](const VectorRef &args) -> VectorRef { return MsRunGraph(graph_id, args, target); });
  MS_EXCEPTION_IF_NULL(result.run);

  result.simu_run = std::make_shared<RunFunc>(
    [graph_id, this](const VectorRef &args) -> VectorRef { return MsSimuRunGraph(graph_id, args); });
  MS_EXCEPTION_IF_NULL(result.simu_run);
  result.graph_id = graph_id;

  graph_id_map_[graph_id] = result;
  if (!pynative_mode) {
    (void)g_ConvertCache.emplace(segment, result);
  }
  return result;
}

// compile set input output
VectorRef MsBackend::MsSimuRunGraph(const GraphId &g, const VectorRef &args) {
  MS_LOG(DEBUG) << "set graph input:" << g;
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
}  // namespace

VectorRef MsBackend::MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target) {
  MS_LOG(DEBUG) << "start ms graph run:" << args.size() << ", g:" << g;
  // Run graph
  std::vector<tensor::TensorPtr> inputs;
  for (const auto &arg : args) {
    PushInputTensor(arg, &inputs);
  }

  VectorRef outputs;
  // Call ms RunGraphAsync or RunOpsInGraph (graphId, input ,output)
  const session::SessionPtr &exe_session = ((target != target_device_ && !target.empty()) ? other_sess_ : target_sess_);
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

void MsBackend::Link(GraphId graph_id) {
  if (graph_id == kInvalidGraphId) {
    graph_id = target_sess_->GetFinalRunGraph();
  }
  target_sess_->BuildGraph(graph_id);
}

MsBackend::MsBackend(const std::string &name, const std::string &target, uint32_t device_id) : Backend(name) {
  convert_fn_ = std::bind(&MsBackend::MsConvert, this, std::placeholders::_1, std::placeholders::_2);
  target_sess_ = session::SessionFactory::Get().Create(target);
  if (target_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed!, please make sure target device:" << target << " is available.";
  }
  target_sess_->Init(device_id);
  target_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
  target_device_ = target;
}

void MsBackend::CreateOtherSession(const std::string &target) {
  if (other_sess_ != nullptr && other_device_ == target) {
    return;
  }
  other_sess_ = session::SessionFactory::Get().Create(target);
  if (other_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed!, please make sure target device:" << target << " is available.";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  other_sess_->Init(device_id);
  other_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
  other_device_ = target;
}

GraphId MsBackend::CompileGraph(NotNull<FuncGraphPtr> fg) { return target_sess_->CompileGraph(fg); }

VectorRef MsBackend::RunGraph(GraphId graph_id, const VectorRef &args) { return MsRunGraph(graph_id, args); }

void MsBackend::ClearSessionGraphs() {
  if (target_sess_ != nullptr) {
    target_sess_->ClearGraph();
  }
}

#ifdef ENABLE_DEBUGGER
void MsBackend::SetDebugger() { target_sess_->SetDebugger(); }
#endif

MindRTBackend::MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id)
    : Backend(backend_name), device_name_(device_name), device_id_(device_id) {
  auto cut_list = compile::GetMsNonlinearOps();
  graph_partition_ = std::make_shared<GraphPartition>(cut_list, backend_name);
}

const ActorInfo &MindRTBackend::CompileGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphPtr root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  // Compile root graph.
  graph_id_to_device_context_.clear();
  control_nodes_.clear();
  CompileGraph(root_graph);

  // Compile sub graphs.
  FuncGraphSet sub_graphs = root_graph->manager()->func_graphs();
  for (auto sub_graph : sub_graphs) {
    if (sub_graph != func_graph && sub_graph != nullptr) {
      CompileGraph(sub_graph);
    }
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(root_graph);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const bool graph_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
  if (graph_mode) {
    // Transform graph to actor DAG, and schedule the actor DAG.
    const auto &actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
    runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  }
  const ActorInfo &actor_info = graph_compiler_info->name_;
  actor_to_graph_compiler_info_.emplace(graph_compiler_info->name_, std::move(graph_compiler_info));
  return actor_info;
}

void MindRTBackend::CompileGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_partition_);

  // Split graph to segments.
  const auto &segments = graph_partition_->Partition(func_graph);
  MS_LOG(INFO) << "Compile graph: " << func_graph->ToString() << ", Split segments size:" << segments.size();

  // Foreach the segments to compile graph.
  for (const auto &segment : segments) {
    MS_EXCEPTION_IF_NULL(segment);
    // Compile the normal nodes, which doesn't contain the cut node.
    if (!segment->is_cut_) {
      if (segment->nodes_.size() == 0) {
        MS_LOG(EXCEPTION) << "The segments size is 0.";
      }
      MS_LOG(INFO) << "Compile normal segment, the first node: " << segment->nodes_[0]->fullname_with_scope();

      // Get and set the device context.
      const auto &cur_device_name = GetCNodeTarget(segment->nodes_[0]);
      const auto &device_context =
        device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({cur_device_name, device_id_});
      device_context->Initialize();
      runtime::GraphCompiler::GetInstance().set_device_context(device_context);

      // Transform nodes to inputs and outputs.
      FuncGraphPtr fg;
      AnfNodePtrList inputs;
      AnfNodePtrList outputs;
      std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(segment->nodes_);

      // Compile graph.
      auto graph_id = runtime::GraphCompiler::GetInstance().CompileGraph(segment->nodes_, outputs);
      graph_id_to_device_context_[graph_id] = device_context;
    } else {
      // Compile the cut node.
      auto cut_node = segment->nodes_[0];
      MS_EXCEPTION_IF_NULL(cut_node);
      MS_LOG(INFO) << "Compile cut segment, the cut node: " << cut_node->fullname_with_scope();
      control_nodes_.push_back(cut_node);
    }
  }
}

const ActorInfo &MindRTBackend::CompileGraph(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                             const std::vector<int64_t> *tensors_mask,
                                             std::vector<tensor::TensorPtr> *input_tensors) {
  static DeviceContext *device_context = nullptr;
  if (!device_context) {
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
    device_context->Initialize();
    runtime::GraphCompiler::GetInstance().set_device_context(device_context);
  }

  bool single_op_cache_hit;
  (void)runtime::GraphCompiler::GetInstance().CompileGraph(op_run_info, graph_info, tensors_mask, input_tensors,
                                                           &single_op_cache_hit);
  if (single_op_cache_hit) {
    return graph_info;
  }

  graph_info_to_device_context_.clear();
  graph_info_to_device_context_[graph_info] = device_context;

  auto graph_compiler_info = ConstructGraphCompilerInfo(tensors_mask, input_tensors);
  const auto actor_set =
    runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info, runtime::GraphExecutionStrategy::kStep);
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  graph_compiler_info->input_tensors_.clear();

  actor_to_graph_compiler_info_.emplace(actor_set->name_, std::move(graph_compiler_info));
  return actor_set->name_;
}

void MindRTBackend::RunGraphBySingleOp(const std::vector<KernelGraphPtr> &graphs,
                                       const std::vector<std::vector<tensor::TensorPtr>> &inputs, VectorRef *outputs) {
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    std::map<AnfNodePtr, size_t> parameter_index;
    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> output_indexes;
    std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;

    runtime::GraphCompiler::GetInstance().GetParamAndOutputIndex(graph, inputs[graph_index], outputs, &parameter_index,
                                                                 &output_indexes);

    for (const auto &kernel : graph->execution_order()) {
      OpRunInfo op_run_info;
      GraphInfo graph_info;
      InputTensorInfo input_tensor_info;
      runtime::GraphCompiler::GetInstance().GetSingleOpInputTensors(kernel, op_output_map, parameter_index,
                                                                    inputs[graph_index], &input_tensor_info);
      runtime::GraphCompiler::GetInstance().GetSingleOpRunInfoAndGraphInfo(kernel, input_tensor_info.input_tensors,
                                                                           &op_run_info, &graph_info);

      const ActorInfo &actor_info =
        CompileGraph(op_run_info, graph_info, &input_tensor_info.input_tensors_mask, &input_tensor_info.input_tensors);
      VectorRef op_outputs =
        RunGraph(actor_info, &input_tensor_info.input_tensors_mask, &input_tensor_info.input_tensors);

      runtime::GraphCompiler::GetInstance().RecoverGraphOutput(kernel, op_outputs, output_indexes, &op_output_map,
                                                               outputs);
    }
  }
}

VectorRef MindRTBackend::RunGraph(const ActorInfo &actor_info, const VectorRef &args) {
  MS_LOG(INFO) << "Run actor begin, actor name: " << actor_info;
  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return VectorRef();
  }

  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info.";
  }
  const auto &graph_compiler_info = *(graph_iter->second.get());
  const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;

  // Transform args to input tensors.
  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    std::vector<tensor::TensorPtr> input_tensor;
    for (const auto &input_node : kernel_graph->input_nodes()) {
      const auto &front_node = kernel_graph->GetFrontAnfByBackendAnf(input_node);
      const auto &iter = std::find(origin_parameters.begin(), origin_parameters.end(), front_node);
      if (iter == origin_parameters.end()) {
        input_tensor.emplace_back(nullptr);
        continue;
      }
      auto position = IntToSize(std::distance(origin_parameters.begin(), iter));
      PushInputTensor(args[position], &input_tensor);
    }
    input_tensors.emplace_back(input_tensor);
  }

  // Run actor DAG.
  VectorRef outputs;
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode) {
    RunGraphBySingleOp(graph_compiler_info.graphs_, input_tensors, &outputs);
    return outputs;
  }

  mindspore::ScopedLongRunning long_running;

  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  runtime::GraphScheduler::GetInstance().PrepareRun(actor_set, graph_compiler_info, input_tensors);
  if (!runtime::GraphScheduler::GetInstance().Run(actor_set)) {
    MS_LOG(EXCEPTION) << "The actor runs failed, actor name: " << actor_set->name_;
  }

  // Sync device stream.
  const auto &first_device_context = graph_compiler_info.device_contexts_[0];
  MS_EXCEPTION_IF_NULL(first_device_context);
  if (!first_device_context->SyncStream()) {
    MS_LOG(EXCEPTION) << "Sync stream failed:" << first_device_context->device_context_key().ToString();
  }
  for (size_t i = 0; i < graph_compiler_info.device_contexts_.size(); ++i) {
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    if ((device_context != first_device_context) && (!device_context->SyncStream())) {
      MS_LOG(EXCEPTION) << "Sync stream failed:" << device_context->device_context_key().ToString();
    }
  }

  // Fetch outputs.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  auto &output_tensors = actor_set->output_actor_->outputs();
  (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(outputs.elements_),
                       [](tensor::TensorPtr &tensor) { return std::move(tensor); });
  MS_LOG(INFO) << "Run actor end, actor name: " << actor_info;
  return outputs;
}

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(root_graph);

  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::string name = "kernel_graph";
  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    graphs.emplace_back(runtime::GraphCompiler::GetInstance().Fetch(graph_id_to_context.first));
    device_contexts.emplace_back(graph_id_to_context.second);
    name.append("_").append(std::to_string(graph_id_to_context.first));
  }

  runtime::KernelMapPosition outputs_order;
  size_t position = 0;
  const auto &outputs = AnfAlgo::GetAllOutput(root_graph->output(), {prim::kPrimTupleGetItem});
  for (const auto &output : outputs) {
    const auto &output_with_index = AnfAlgo::VisitKernelWithReturnType(output, 0, true);
    MS_EXCEPTION_IF_NULL(output_with_index.first);
    outputs_order.emplace(output_with_index, position++);
  }

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes_,
                                             root_graph->parameters(), outputs_order, name);
}

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(
  const std::vector<int64_t> *tensors_mask, const std::vector<tensor::TensorPtr> *input_tensors) {
  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  runtime::KernelMapPosition outputs_order;
  size_t position = 0;
  std::string name;

  for (const auto &graph_info_to_context : graph_info_to_device_context_) {
    const auto &graph = runtime::GraphCompiler::GetInstance().Fetch(graph_info_to_context.first);
    graphs.emplace_back(graph);
    device_contexts.emplace_back(graph_info_to_context.second);
    name.append(graph_info_to_context.first);

    const auto &outputs = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
    for (const auto &output : outputs) {
      const auto &output_with_index = AnfAlgo::VisitKernelWithReturnType(output, 0, false);
      MS_EXCEPTION_IF_NULL(output_with_index.first);
      outputs_order.emplace(output_with_index, position++);
    }
  }
  std::vector<std::vector<int64_t> *> tensors_mask_list(1, const_cast<std::vector<int64_t> *>(tensors_mask));
  std::vector<std::vector<TensorPtr> *> input_tensors_list(1,
                                                           const_cast<std::vector<tensor::TensorPtr> *>(input_tensors));

  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask_list, input_tensors_list,
                                             std::vector<AnfNodePtr>(), std::vector<AnfNodePtr>(), outputs_order, name);
}

VectorRef MindRTBackend::RunGraph(const ActorInfo &actor_info, const std::vector<int64_t> *tensors_mask,
                                  const std::vector<tensor::TensorPtr> *input_tensors) {
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info.";
  }
  const auto &graph_compiler_info = *(graph_iter->second);

  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  MS_EXCEPTION_IF_NULL(actor_set);

  // Erase value node tensor.
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  if (input_tensors->size() != tensors_mask->size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors->size() << " should be equal to tensors mask size "
                      << tensors_mask->size();
  }
  for (size_t index = 0; index < tensors_mask->size(); ++index) {
    if (tensors_mask->at(index) != kValueNodeTensorMask) {
      tensors_without_value_node.emplace_back(input_tensors->at(index));
    }
  }

  mindspore::ScopedLongRunning long_running;
  runtime::GraphScheduler::GetInstance().PrepareRun(actor_set, graph_compiler_info, {tensors_without_value_node});
  if (!runtime::GraphScheduler::GetInstance().Run(actor_set, runtime::GraphExecutionStrategy::kStep, input_tensors)) {
    MS_LOG(EXCEPTION) << "The actor runs failed, actor name: " << actor_set->name_;
  }

  // Fetch outputs.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  auto &output_tensors = actor_set->output_actor_->outputs();
  VectorRef outputs;
  (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(outputs.elements_),
                       [](tensor::TensorPtr &tensor) { return std::move(tensor); });
  return outputs;
}
}  // namespace compile
}  // namespace mindspore
