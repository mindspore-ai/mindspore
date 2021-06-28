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
#include "backend/optimizer/common/helper.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/parse/data_converter.h"
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
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
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

// Insert the front_node related tensor in the input_tensor.
void PushTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                std::vector<tensor::TensorPtr> *input_tensor) {
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  if (iter == parameters.end()) {
    (*input_tensor).emplace_back(nullptr);
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

void ClearDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
      auto old_device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
      MS_EXCEPTION_IF_NULL(old_device_address);

      auto new_device_tensor = device_context->CreateDeviceAddress(
        nullptr, old_device_address->GetSize(), old_device_address->format(), old_device_address->type_id());
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      new_device_tensor->set_original_ref_count(old_device_address->original_ref_count());
      new_device_tensor->ResetRefCount();
      AnfAlgo::SetOutputAddr(new_device_tensor, 0, node.get());
    }
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

  runtime::GraphScheduler::GetInstance().Initialize();
}

const ActorInfo &MindRTBackend::CompileGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(func_graph);
  root_graph_ = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph_);
  // Register a summary callback function, which is called in the final stages of summary.
  graph_compiler_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);

  // Compile root graph.
  graph_id_to_device_context_.clear();
  control_nodes_.clear();
  CompileGraph(root_graph_);

  // Compile sub graphs.
  FuncGraphSet sub_graphs = root_graph_->manager()->func_graphs();
  for (auto sub_graph : sub_graphs) {
    if (sub_graph != func_graph && sub_graph != nullptr) {
      CompileGraph(sub_graph);
    }
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(root_graph_);

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
  MS_EXCEPTION_IF_NULL(graph_compiler_);

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

      // Get the device context.
      const auto &cur_device_name = GetCNodeTarget(segment->nodes_[0]);
      const auto &device_context =
        device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({cur_device_name, device_id_});
      device_context->Initialize();

      // Transform nodes to inputs and outputs.
      FuncGraphPtr fg;
      AnfNodePtrList inputs;
      AnfNodePtrList outputs;
      std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(segment->nodes_);

      // Compile graph.
      auto graph_id = graph_compiler_->CompileGraph(segment->nodes_, outputs, device_context);
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
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  // Get the device context.
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  device_context->Initialize();

  bool single_op_cache_hit;
  auto graph_id = graph_compiler_->CompileGraph(op_run_info, graph_info, tensors_mask, input_tensors,
                                                &single_op_cache_hit, device_context);
  // The actor set name: graph_id + single operator name.
  std::string actor_info = std::to_string(graph_id) + "_" + op_run_info.op_name;
  if (single_op_cache_hit) {
    auto iter = actor_to_graph_compiler_info_.find(actor_info);
    if (iter == actor_to_graph_compiler_info_.end()) {
      MS_LOG(EXCEPTION) << "Can not find graph compiler info for actor set: " << actor_info;
    }
    return iter->first;
  }

  graph_info_to_device_context_.clear();
  graph_info_to_device_context_[graph_info] = device_context;

  auto graph_compiler_info = ConstructGraphCompilerInfo(actor_info, tensors_mask, input_tensors);
  const auto actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  graph_compiler_info->input_tensors_.clear();

  auto ret = actor_to_graph_compiler_info_.emplace(actor_info, std::move(graph_compiler_info));
  return ret.first->first;
}

namespace {
void PlantTensorTupleToVector(const py::tuple &tuple_inputs, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  for (const auto &input_object : tuple_inputs) {
    if (!py::isinstance<tensor::Tensor>(input_object)) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    auto tensor = py::cast<tensor::TensorPtr>(input_object);
    MS_EXCEPTION_IF_NULL(tensor);
    tensors->emplace_back(tensor);
  }
}

void ConvertValueTupleToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  ValuePtr input_value = parse::data_converter::PyDataToValue(input_object);
  MS_EXCEPTION_IF_NULL(input_value);
  if (!input_value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "The input object is not a value tuple!";
  }

  auto value_tuple = input_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  tensors->emplace_back(tensor_ptr);
}

void ConvertMultiPyObjectToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  if (!py::isinstance<py::tuple>(input_object)) {
    MS_LOG(EXCEPTION) << "The input should be a tuple!";
  }

  auto tuple_inputs = py::cast<py::tuple>(input_object);
  if (tuple_inputs.empty()) {
    MS_LOG(EXCEPTION) << "The size of input list or tuple is 0!";
  }

  auto inputs = py::cast<py::tuple>(input_object);
  if (py::isinstance<tensor::Tensor>(inputs[0])) {
    PlantTensorTupleToVector(inputs, tensors);
  } else {
    ConvertValueTupleToTensor(input_object, tensors);
  }
}

void RunControlOperator(const KernelGraphPtr &graph, const AnfNodePtr &kernel, std::vector<TensorPtr> *input_tensors,
                        VectorRef *op_outputs) {
  AnfNodePtr front_node = graph->GetFrontAnfByBackendAnf(kernel);
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The front node of bprop_cut is not CNode";
  }

  CNodePtr cnode = front_node->cast<CNodePtr>();
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
  if (prim->name() == kBpropCutOpName) {
    VectorRef args;
    (void)std::transform(input_tensors->begin(), input_tensors->end(), std::back_inserter(args.elements_),
                         [](tensor::TensorPtr &tensor) { return std::move(tensor); });

    BaseRef out = prim->RunHookFunction(args);

    if (utils::isa<PyObjectRef>(out)) {
      PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
      auto out_py_tuple = py_ref.object_;
      std::vector<tensor::TensorPtr> output_tensors;
      ConvertMultiPyObjectToTensor(out_py_tuple, &output_tensors);
      (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(op_outputs->elements_),
                           [](tensor::TensorPtr &tensor) { return std::move(tensor); });
    }
  }
}
}  // namespace

void MindRTBackend::RunGraphBySingleOp(const std::vector<KernelGraphPtr> &graphs,
                                       const std::vector<std::vector<tensor::TensorPtr>> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;

    std::map<AnfNodePtr, size_t> parameter_index;
    GraphOutputInfo graph_output_info;
    graph_output_info.graph_outputs = outputs;
    graph_compiler_->GetParamAndOutputIndex(graph, inputs[graph_index], outputs, &parameter_index,
                                            &graph_output_info.output_indexes);

    std::map<KernelWithIndex, size_t> cnode_ref_count;
    graph_compiler_->CalculateRefCount(graph, &cnode_ref_count);

    // Clear bucket resources every step
    if (graph->is_bprop()) {
      graph_compiler_->ClearAllBucket(graph->graph_id());
    }

    for (const auto &kernel : graph->execution_order()) {
      OpRunInfo op_run_info;
      GraphInfo graph_info;
      InputTensorInfo input_tensor_info;
      graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index],
                                               &input_tensor_info);
      graph_compiler_->GetSingleOpRunInfoAndGraphInfo(kernel, input_tensor_info.input_tensors, &op_run_info,
                                                      &graph_info);

      VectorRef op_outputs;
      if (!AnfAlgo::IsControlOpExecInBackend(kernel)) {
        const ActorInfo &actor_info = CompileGraph(op_run_info, graph_info, &input_tensor_info.input_tensors_mask,
                                                   &input_tensor_info.input_tensors);
        RunGraph(actor_info, &op_run_info, &input_tensor_info.input_tensors_mask, &input_tensor_info.input_tensors,
                 &op_outputs);
      } else {
        RunControlOperator(graph, kernel, &input_tensor_info.input_tensors, &op_outputs);
      }

      graph_compiler_->UpdateRefCount(input_tensor_info.input_kernel, &cnode_ref_count, &op_output_map);

      graph_output_info.graph_output_tensors.clear();
      graph_compiler_->RecoverGraphOutput(kernel, op_outputs, cnode_ref_count, &op_output_map, &graph_output_info);

      // Save grad node to Bucket
      if (graph->is_bprop() && (!AnfAlgo::IsControlOpExecInBackend(kernel))) {
        graph_compiler_->AddGradAddrToBucket(graph->graph_id(), graph_output_info.graph_output_tensors);
      }
    }
  }
}

void MindRTBackend::RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Run actor begin, actor name: " << actor_info;
  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return;
  }

  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info.";
  }
  const auto &graph_compiler_info = *(graph_iter->second.get());
  const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;

  // Transform args to input tensors.
  // Input tensors of the graph.
  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    std::vector<tensor::TensorPtr> input_tensor;
    for (const auto &input_node : kernel_graph->input_nodes()) {
      const auto &front_node = kernel_graph->GetFrontAnfByBackendAnf(input_node);
      PushTensor(args, origin_parameters, front_node, &input_tensor);
    }
    input_tensors.emplace_back(input_tensor);
  }

  // Input tensors of the control node.
  std::vector<tensor::TensorPtr> input_tensor;

  // Get inputs of control node which come from the host actor.
  const auto &control_node_parameters = graph_compiler_info.control_node_parser_->GetControlNodeParameter();
  for (const auto &parameter : control_node_parameters) {
    PushTensor(args, origin_parameters, parameter, &input_tensor);
  }
  input_tensors.emplace_back(input_tensor);

  // Run in the pynative mode.
  MS_EXCEPTION_IF_NULL(outputs);
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode) {
    RunGraphBySingleOp(graph_compiler_info.graphs_, input_tensors, outputs);
    return;
  }

  mindspore::ScopedLongRunning long_running;
  // Run actor DAG.
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  runtime::GraphScheduler::GetInstance().PrepareRun(actor_set, graph_compiler_info, input_tensors);
// Debugger pre-execute graph.
#ifdef ENABLE_DEBUGGER
  if (Debugger::GetInstance()->DebuggerBackendEnabled()) {
    Debugger::GetInstance()->PreExecuteGraphDebugger(graph_compiler_info.graphs_);
  }
#endif
  if (!runtime::GraphScheduler::GetInstance().Run(actor_set)) {
    MS_LOG(EXCEPTION) << "The actor runs failed, actor name: " << actor_set->name_;
  }
// Debugger post-execute graph.
#ifdef ENABLE_DEBUGGER
  if (Debugger::GetInstance()->DebuggerBackendEnabled()) {
    Debugger::GetInstance()->PostExecuteGraphDebugger(graph_compiler_info.graphs_);
  }
#endif

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
  if (output_tensors.size() > 0) {
    size_t output_position = 0;
    ConstructOutputs(root_graph_->output(), output_tensors, &output_position, outputs);
  }
  MS_LOG(INFO) << "Run actor end, actor name: " << actor_info;

  graph_compiler_->Summary(graph_compiler_info.graphs_);

  // Update device address for output node of graph.
  actor_set->output_actor_->UpdateOutputDeviceAddress();
}

void MindRTBackend::ConstructOutputs(const AnfNodePtr &output_node,
                                     const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                     VectorRef *outputs) {
  // The makeTuple node need expand and recurse.
  if (AnfAlgo::CheckPrimitiveType(output_node, prim::kPrimMakeTuple)) {
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

  // The empty value node return the empty VectorRef.
  if (output_node->isa<ValueNode>()) {
    auto value = output_node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueTuple>() && (value->cast<ValueTuplePtr>()->size() == 0)) {
      outputs->emplace_back(VectorRef());
      return;
    }
  }

  auto outputs_num = AnfAlgo::GetOutputTensorNum(output_node);
  auto &output_abstract = output_node->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  // Wrap output to VectorRef if the output is tuple.
  if (output_abstract->isa<abstract::AbstractTuple>()) {
    VectorRef output_tuple;
    for (size_t i = 0; i < outputs_num; ++i) {
      output_tuple.emplace_back(std::move(output_tensors[*output_position]));
      ++(*output_position);
    }
    outputs->emplace_back(std::move(output_tuple));
  } else {
    for (size_t i = 0; i < outputs_num; ++i) {
      outputs->emplace_back(std::move(output_tensors[*output_position]));
      ++(*output_position);
    }
  }
}

#ifdef ENABLE_DEBUGGER
void MindRTBackend::SetDebugger() {
  auto debugger_ = Debugger::GetInstance();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  debugger_->Init(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
}
#endif

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::string name = "kernel_graph";
  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    graphs.emplace_back(graph_compiler_->Fetch(graph_id_to_context.first));
    device_contexts.emplace_back(graph_id_to_context.second);
    name.append("_").append(std::to_string(graph_id_to_context.first));
  }

  auto parser = std::make_shared<ControlNodeParser>();
  parser->Parse(control_nodes_, graphs, device_contexts, root_graph);

  // Get all the outputs. In control flow, there may be multiple branch output.
  runtime::KernelMapPosition outputs_order;
  size_t outputs_num = 0;
  const auto &all_branch_output = parser->FetchAllBranchOutputs(root_graph);
  for (int j = 0; j < SizeToInt(all_branch_output.size()); ++j) {
    // In general, there is only one output branch, and the branch id is 0 at this time. In the control flow,
    // there are multi-branch output scenarios. Different branches may have different weight nodes. When output
    // actor run, the corresponding weight node needs to be obtained according to different branches. Therefore,
    // the branch of the output nodes needs to be recorded.
    const int branch_id = ((all_branch_output.size() == 1 ? runtime::kMainBranchID : (j + runtime::kSubBranchStartID)));
    const auto &branch_output = all_branch_output[j];
    size_t position = 0;
    auto outputs = AnfAlgo::GetAllOutputWithIndex(branch_output);
    outputs_num = outputs.size();
    for (const auto &output : outputs) {
      if (outputs_order.count(output) == 0) {
        outputs_order[output] = {branch_id, {position++}};
      } else {
        outputs_order[output].second.emplace_back(position++);
      }
    }
  }

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes_,
                                             root_graph->parameters(), parser, outputs_order, outputs_num, name,
                                             runtime::GraphExecutionStrategy::kPipeline);
}

std::unique_ptr<GraphCompilerInfo> MindRTBackend::ConstructGraphCompilerInfo(
  const ActorInfo &actor_info, const std::vector<int64_t> *tensors_mask,
  const std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  runtime::KernelMapPosition outputs_order;
  size_t position = 0;

  for (const auto &graph_info_to_context : graph_info_to_device_context_) {
    const auto &graph = graph_compiler_->Fetch(graph_info_to_context.first);
    graphs.emplace_back(graph);
    device_contexts.emplace_back(graph_info_to_context.second);

    auto outputs = AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output : outputs) {
      if (outputs_order.count(output) == 0) {
        outputs_order[output] = {runtime::kMainBranchID, {position++}};
      } else {
        outputs_order[output].second.emplace_back(position++);
      }
    }
  }

  std::vector<std::vector<int64_t> *> tensors_mask_list(1, const_cast<std::vector<int64_t> *>(tensors_mask));
  std::vector<std::vector<TensorPtr> *> input_tensors_list(1,
                                                           const_cast<std::vector<tensor::TensorPtr> *>(input_tensors));
  auto parser = std::make_shared<ControlNodeParser>();
  return std::make_unique<GraphCompilerInfo>(graphs, device_contexts, tensors_mask_list, input_tensors_list,
                                             std::vector<AnfNodePtr>(), std::vector<AnfNodePtr>(), parser,
                                             outputs_order, outputs_order.size(), actor_info,
                                             runtime::GraphExecutionStrategy::kStep);
}

void MindRTBackend::RunGraph(const ActorInfo &actor_info, OpRunInfo *op_run_info,
                             const std::vector<int64_t> *tensors_mask,
                             const std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs) {
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

  for (auto &tensor : tensors_without_value_node) {
    if (tensor->NeedWaitDevice()) {
      tensor->WaitDevice();
    }
  }

  runtime::GraphScheduler::GetInstance().PrepareRun(actor_set, graph_compiler_info, {tensors_without_value_node});
  if (!runtime::GraphScheduler::GetInstance().Run(actor_set, runtime::GraphExecutionStrategy::kStep, input_tensors)) {
    MS_LOG(EXCEPTION) << "The actor runs failed, actor name: " << actor_set->name_;
  }

  // Fetch outputs.
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  auto &output_tensors = actor_set->output_actor_->outputs();
  (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(outputs->elements_),
                       [](tensor::TensorPtr &tensor) { return std::move(tensor); });

  // Update output abstract of dynamic op to op_run_info
  if (op_run_info->is_dynamic_shape) {
    UpdateOutputAbstract(graph_compiler_info.graphs_.front(), op_run_info);
  }

  // Release the kernel resource.
  const auto &graph = graph_compiler_info.graphs_.front();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernel = graph->execution_order().front();
  MS_EXCEPTION_IF_NULL(kernel);
  if (kOpCacheBlackList.find(AnfAlgo::GetCNodeName(kernel)) != kOpCacheBlackList.end()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    if (kernel_mod) {
      kernel_mod->ReleaseResource();
    }
  }

  // Update device address for output node of graph.
  actor_set->output_actor_->UpdateOutputDeviceAddress();
  ClearDeviceAddress(graph_compiler_info.graphs_.front(), graph_compiler_info.device_contexts_.front());
}
}  // namespace compile
}  // namespace mindspore
