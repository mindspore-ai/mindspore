/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <stack>
#include <unordered_map>
#include "include/common/utils/parallel_context.h"
#include "backend/graph_compiler/transform.h"
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "backend/common/optimizer/helper.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/parse/data_converter.h"
#include "ir/anf.h"
#include "pybind_api/ir/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/callbacks.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/pynative/run_op_helper.h"
#include "runtime/pynative/graph_adapter.h"
#include "distributed/recovery/recovery_context.h"
#include "include/common/utils/scoped_long_running.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/ps_context.h"
#endif

#include "runtime/device/device_address_utils.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

namespace mindspore {
namespace compile {
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
  for (const auto &pre_segment : segment->pre_segments_) {
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
std::vector<tensor::TensorPtr> GetTensorWithoutValueMask(const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  const auto &input_tensors = op_run_info->base_op_run_info.input_tensor;
  const auto &tensors_mask = op_run_info->base_op_run_info.input_mask;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t index = 0; index < tensors_mask.size(); ++index) {
    if (tensors_mask.at(index) != kValueNodeTensorMask) {
      (void)tensors_without_value_node.emplace_back(input_tensors.at(index));
    }
  }
  return tensors_without_value_node;
}

device::DeviceAddressPtr CloneEmptyDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                 const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(device_context);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, old_device_address->GetSize(), old_device_address->format(), old_device_address->type_id(),
    old_device_address->host_shape());
  MS_EXCEPTION_IF_NULL(new_device_address);
  new_device_address->set_original_ref_count(old_device_address->original_ref_count());
  new_device_address->ResetRefCount();
  auto node = old_device_address->GetNodeIndex();
  new_device_address->SetNodeIndex(node.first, node.second);
  return new_device_address;
}

void ClearGraphDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context, bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : graph->execution_order()) {
    auto output_address_num = AnfAlgo::GetOutputAddressNum(node);
    // Clear old output device address of kernel
    for (size_t i = 0; i < output_address_num; ++i) {
      if (!AnfAlgo::OutputAddrExist(node, i, false)) {
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
      if (device_address == nullptr) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(device_context);
      auto new_device_address = CloneEmptyDeviceAddress(device_address, device_context);
      if (is_gradient_out) {
        new_device_address->set_from_persistent_mem(true);
      }
      AnfAlgo::SetOutputAddr(new_device_address, i, node.get());
    }

    // Clear old workspace device address of kernel
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      if (!AnfAlgo::WorkspaceAddrExist(node, i)) {
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableWorkspaceAddr(node, i);
      auto new_device_address = CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetWorkspaceAddr(new_device_address, i, node.get());
    }
  }
}

void ClearGraphDeviceAddressDynamic(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                    bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : graph->execution_order()) {
    auto output_address_num = AnfAlgo::GetOutputAddressNum(node);
    // Clear old output device address of kernel
    for (size_t i = 0; i < output_address_num; ++i) {
      AnfAlgo::SetOutputAddr(nullptr, i, node.get());
    }

    // Clear old workspace device address of kernel
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      AnfAlgo::SetWorkspaceAddr(nullptr, i, node.get());
    }
  }
}

void ClearInputDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
      if (device_address == nullptr) {
        continue;
      }
      auto new_device_address = CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetOutputAddr(new_device_address, 0, node.get());
    }
  }
}

void ClearInputDeviceAddressDynamic(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      AnfAlgo::SetOutputAddr(nullptr, 0, node.get());
    }
  }
}

bool OpInBlackList(const session::BackendOpRunInfoPtr &op_run_info) {
  return IsOneOfCacheBlackList(op_run_info->base_op_run_info.op_name);
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

#if defined(__linux__) && defined(WITH_BACKEND)
  // If in PS mode, must use sync mode to run graph in case that the weights on server are not updated in the last step.
  if (ps::PSContext::instance()->is_ps_mode()) {
    exe_session->RunGraph(g, inputs, &outputs);
    return outputs;
  }
#endif

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

namespace {
void GetControlOpInput(const std::shared_ptr<GraphCompiler> &graph_compiler, const CNodePtr &front_cnode,
                       const CNodePtr &backend_cnode, const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                       const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                       VectorRef *args) {
  MS_EXCEPTION_IF_NULL(front_cnode);
  MS_EXCEPTION_IF_NULL(backend_cnode);
  MS_EXCEPTION_IF_NULL(graph_compiler);
  MS_EXCEPTION_IF_NULL(args);
  size_t front_index = 0;     // Point to front end cnode
  size_t back_index = 0;      // Point to backend end cnode
  size_t args_tuple_num = 0;  // Record the input num of maketuple cnode
  std::vector<ValuePtr> args_tuple;
  auto front_size = front_cnode->inputs().size();
  auto back_size = backend_cnode->inputs().size();
  while (front_index + 1 < front_size && back_index + 1 < back_size) {
    AnfNodePtr input_node = nullptr;
    if (args_tuple_num != 0) {
      input_node = backend_cnode->input(back_index + 1);
    } else {
      input_node = front_cnode->input(front_index + 1);
      if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
        // Hook multi-input or multi-output.
        MS_LOG(DEBUG) << "The input node of hook op: " << input_node->DebugString() << " is a make tuple node.";
        auto make_tuple = input_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple);
        args_tuple_num = make_tuple->inputs().size() - 1;
        continue;
      } else if (input_node->isa<Parameter>()) {
        auto param = input_node->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param);
        auto abs = param->abstract();
        MS_EXCEPTION_IF_NULL(abs);
        if (abs->isa<abstract::AbstractTuple>() && !abs->isa<abstract::AbstractSparseTensor>()) {
          auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
          MS_EXCEPTION_IF_NULL(abs_tuple);
          args_tuple_num = abs_tuple->elements().size();
          continue;
        }
      }
    }
    // Hook single-input or single-output.
    auto real_input = common::AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(real_input);
    ValuePtr value = nullptr;
    if (!real_input->isa<ValueNode>()) {
      if (real_input->abstract() != nullptr && real_input->abstract()->isa<abstract::AbstractSparseTensor>()) {
        value = TensorListToSparseTensor(real_input->abstract(), graph_inputs);
      } else {
        value = graph_compiler->GetSingleOpInputTensorByIndex(backend_cnode, op_output_map, parameter_index,
                                                              graph_inputs, input_tensor_info, back_index);
      }
      MS_EXCEPTION_IF_NULL(value);
      ++back_index;
    } else {
      const auto &value_node = real_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      if (value->isa<ValueSequence>()) {
        const auto &value_sequeue = value->cast<ValueSequencePtr>();
        MS_EXCEPTION_IF_NULL(value_sequeue);
        back_index += value_sequeue->size();
      } else {
        ++back_index;
      }
    }
    if (args_tuple_num != 0) {
      (void)args_tuple.emplace_back(value);
      if (args_tuple.size() == args_tuple_num) {
        value = std::make_shared<ValueTuple>(args_tuple);
        args_tuple_num = 0;
        args_tuple.clear();
      }
    }
    if (args_tuple_num == 0) {
      args->emplace_back(value);
      front_index++;
    }
  }
}

void ConvertPyObjectToTensor(const py::object &input_object, std::vector<ValuePtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  ValuePtr tensor_ptr = nullptr;
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
  } else if (py::isinstance<tensor::CSRTensor>(input_object)) {
    tensor_ptr = py::cast<tensor::CSRTensorPtr>(input_object);
  } else if (py::isinstance<tensor::COOTensor>(input_object)) {
    tensor_ptr = py::cast<tensor::COOTensorPtr>(input_object);
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
    py::gil_scoped_acquire acquire;
    BaseRef out = python_adapter::PyAdapterCallback::RunPrimitivePyHookFunction(prim, args);
    // Convert pyobject output to tensor.
    if (utils::isa<PyObjectRef>(out)) {
      PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
      auto out_py_tuple = py_ref.object_;
      std::vector<ValuePtr> output_tensors;
      ConvertPyObjectToTensor(out_py_tuple, &output_tensors);
      (void)std::transform(output_tensors.begin(), output_tensors.end(), std::back_inserter(op_outputs->elements_),
                           [](ValuePtr &tensor) { return std::move(tensor); });
    }
  }
}
void UpdateOutputAbstract(const KernelGraphPtr &kernel_graph, const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::GetCNodeName(kernel) == op_run_info->base_op_run_info.op_name) {
      op_run_info->base_op_run_info.abstract = kernel->abstract();
    }
  }
}

TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(output_node);
  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
  const auto &shape = common::AnfAlgo::GetOutputInferShape(output_node, output_index);
  auto tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));

  // Put device tensor into host tensor.
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->SetNodeIndex(output_node, output_index);
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
}  // namespace

void MindRTBackend::RunGraphByActors(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                                     const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Start";
  WaitTaskFinish();
  auto inputs = GetRunGraphInputs(graph_compiler_info, args);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto graphs = graph_compiler_info.graphs_;
  auto device_contexts = graph_compiler_info.device_contexts_;
  if (graphs.size() > inputs.size()) {
    MS_LOG(EXCEPTION) << "The actor_set " << actor_info << " graphs size " << graphs.size()
                      << " should less than or equal to inputs size " << inputs.size();
  }
  if (device_contexts.size() != graphs.size()) {
    MS_LOG(EXCEPTION) << "Graphs size " << graphs.size() << " is not equal to device_contexts size "
                      << device_contexts.size();
  }

  // KernelByKernel: The size of control_nodes is at least 1 since there is return node in the graph.
  // GraphMode: No control nodes.
  bool no_control_flow = control_nodes_.size() <= 1 && graphs.size() == 1;

  auto actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  if (actor_set == nullptr) {
    // Need to compile graph for the first step.
    for (size_t i = 0; i < graphs.size(); ++i) {
      const auto &graph = graphs[i];
      MS_EXCEPTION_IF_NULL(graph);
      graph->set_flag(kFlagPyNativeRunInGraph, true);
      graph->set_flag(kFlagIsPynativeBpropGraph, root_graph_->has_flag(kFlagIsPynativeBpropGraph));

      if (no_control_flow) {
        MS_LOG(INFO) << "Replace parameter format";
        // The input tensors of heterogeneous graphs or control flow graphs are null.
        // Need to get tensor after ParseControlNodes.
        pynative::GraphAdapter::ReplaceGraphParameterProperties(graph, inputs.at(i), device_contexts[i]);
      }
      (void)graph_compiler_->CompileGraphImpl(graph, device_contexts[i]);
      pynative::GraphAdapter::RemoveUnusedValueNodes(graph);
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, graph->front_outputs());
      // Clear front outputs after the outputs is cached.
      graph->set_front_outputs({});
      AnfAlgo::UpdateGraphValidRefPair(graph);
      pynative::GraphAdapter::SensTensorToDevice(graph, device_contexts[i]);
    }

    ParseControlNodes(graph_compiler_info);
    actor_set = runtime::GraphScheduler::GetInstance().Transform(graph_compiler_info);
    MS_EXCEPTION_IF_NULL(actor_set);
    constexpr auto kKernelActorThreshold = 5000;
    // Turning off multithreading may cause stack overflow in control flow scenarios.
    if (no_control_flow && actor_set->kernel_actors_.size() < kKernelActorThreshold) {
      // Multithreading can cause spikes in memory usage and performance fluctuations.
      actor_set->is_multi_thread_execution_ = false;
      MS_LOG(INFO) << "Actor Multithreading is turned off!";
    }
    runtime::GraphScheduler::GetInstance().Schedule(actor_set);

    for (auto &graph : graphs) {
      pynative::GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(graph);
      pynative::GraphAdapter::GenerateRefCountForBpropValueNode(graph);
    }
  }

  if (root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    for (size_t i = 0; i < graphs.size(); ++i) {
      pynative::GraphAdapter::UpdateForwardOutputInBpropGraph(graphs[i], device_contexts[i], no_control_flow);
      pynative::GraphAdapter::UpdateDynamicValueNodeAbstract(graphs[i]);
    }
  }

  std::vector<std::vector<tensor::TensorPtr>> input_tensors = GetRunGraphInputs(graph_compiler_info, args);
  pynative::GraphAdapter::HandleHeterogeneousTensors(input_tensors, device_contexts);

  // Release GIL and run actor DAG.
  mindspore::ScopedLongRunning long_running;
  runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors);

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  ConstructOutputs(actor_set, outputs, root_graph_);

  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
}

void MindRTBackend::RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (!IsValueNode<FuncGraph>(kernel)) {
    MS_LOG(EXCEPTION) << "kernel:" << kernel->ToString() << ", is not func graph.";
  }
  auto func_graph = GetValueNode<FuncGraphPtr>(kernel);
  func_graph->set_flag(kFlagIsPynativeBpropGraph, true);

  auto old_root_graph = root_graph_;
  auto actor_info = CompileGraphs(func_graph);

  MS_LOG(INFO) << "Status record: start run actor: " << actor_info;
  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info, actor_info:" << actor_info;
  }
  MS_EXCEPTION_IF_NULL(graph_iter->second);
  const auto &graph_compiler_info = *(graph_iter->second);

  RunGraphByActors(actor_info, graph_compiler_info, args, outputs);
  root_graph_ = old_root_graph;
}

void MindRTBackend::RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                       VectorRef *outputs) {
  WaitTaskFinish();
  auto &op_executor = runtime::OpExecutor::GetInstance();
  op_executor.Register([this]() { BatchBuildCallback(); });

  MS_LOG(INFO) << "Start";
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  const auto &graphs = graph_compiler_info.graphs_;
  auto inputs = GetRunGraphInputs(graph_compiler_info, args);
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

    MS_EXCEPTION_IF_NULL(root_graph_);
    if (root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
      graph_compiler_->CalculateForwardOpOutputCount(graph, inputs[graph_index], &forward_op_output_tensor_id_);
    }

    bool graph_use_dynamic_process = root_graph_->has_flag(kFlagUseDynamicShapeProcess);
    py::gil_scoped_release release;
    for (const auto &kernel : graph->execution_order()) {
      InputTensorInfo input_tensor_info;
      VectorRef op_outputs;
      if (common::AnfAlgo::IsControlOpExecInBackend(kernel)) {
        WaitTaskFinish();
        RunControlOperator(graph_compiler_, graph, kernel, op_output_map, parameter_index, inputs[graph_index],
                           &input_tensor_info, &op_outputs);
        // Execute remaining lazy tasks before PyNative hook exit.
        WaitTaskFinish();
      } else if (enable_backend_dynamic_detect_ && common::AnfAlgo::HasNodeAttr(kAttrMsFunctionControl, kernel)) {
        WaitTaskFinish();
        graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index],
                                                 &input_tensor_info);
        VectorRef input_args;
        (void)std::transform(input_tensor_info.input_tensors.begin(), input_tensor_info.input_tensors.end(),
                             std::back_inserter(input_args.elements_),
                             [](tensor::TensorPtr &tensor) { return std::move(tensor); });

        RunMsGradGraph(kernel, input_args, &op_outputs);
        WaitTaskFinish();
      } else {
        session::BackendOpRunInfoPtr op_run_info;
        GraphInfo graph_info;
        auto op_use_dynamic_process =
          graph_use_dynamic_process &&
          pynative::OpCompiler::GetInstance().NeedEnableDynamicProcess(common::AnfAlgo::GetCNodeName(kernel));
        if (op_use_dynamic_process) {
          common::AnfAlgo::SetNodeAttr(kAttrMutableKernel, MakeValue(true), kernel);
        }
        graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index],
                                                 &input_tensor_info);
        graph_compiler_->GetSingleOpRunInfoAndGraphInfo(kernel, input_tensor_info, op_use_dynamic_process, &op_run_info,
                                                        &graph_info, &graph_output_info);
        if (op_use_dynamic_process) {
          op_run_info->op_prim->AddAttr(kAttrMutableKernel, MakeValue(true));
          op_run_info->op_prim->AddAttr(kAttrInputIsDynamicShape, MakeValue(true));
          op_run_info->op_prim->AddAttr(kAttrOutputIsDynamicShape, MakeValue(true));
          RunOpDynamic(op_run_info, &op_outputs);
        } else {
          RunOp(op_run_info, &op_outputs);
        }
      }

      graph_compiler_->UpdateRefCount(input_tensor_info.input_kernel, &cnode_ref_count, &op_output_map);

      graph_output_info.graph_output_tensors.clear();
      graph_compiler_->RecoverGraphOutput(kernel, op_outputs, cnode_ref_count, &op_output_map, &graph_output_info);
    }
    WaitTaskFinish();
  }
  if (is_dynamic_ || root_graph_->has_flag(kFlagUseDynamicShapeProcess)) {
    ClearResource();
  }
  MS_LOG(INFO) << "End";
}

void MindRTBackend::RunGraphByCondition(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                                        const VectorRef &args, VectorRef *outputs) {
  bool enable_run_graph_by_single_op =
    std::any_of(graph_compiler_info.graphs_.begin(), graph_compiler_info.graphs_.end(),
                [](const KernelGraphPtr &graph) { return graph->has_flag(kFlagEnableRunGraphBySingleOp); });
  MS_EXCEPTION_IF_NULL(root_graph_);
  bool is_dynamic = true;

  std::string func_graph_cell_id;
  if (enable_backend_dynamic_detect_ && root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    func_graph_cell_id = GetValue<std::string>(root_graph_->get_attr(kAttrFuncGraphCellId));
    auto graph_iter = func_graph_dynamic_infos_.find(func_graph_cell_id);
    if (graph_iter == func_graph_dynamic_infos_.end()) {
      MS_LOG(EXCEPTION) << "cell id[" << func_graph_cell_id << "] is not found in func_graph_dynamic_infos_";
    }
    is_dynamic = graph_iter->second.is_dynamic;
  }

  if (enable_run_graph_by_single_op || root_graph_->has_flag(kFlagIsDynamicStructure) ||
      (enable_backend_dynamic_detect_ && root_graph_->has_flag(kFlagIsPynativeBpropGraph) && is_dynamic) ||
      root_graph_->has_flag(kFlagUseDynamicShapeProcess)) {
    RunGraphBySingleOp(graph_compiler_info, args, outputs);
  } else {
    RunGraphByActors(actor_info, graph_compiler_info, args, outputs);
  }

  if (enable_backend_dynamic_detect_ && root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    auto graph_iter = func_graph_dynamic_infos_.find(func_graph_cell_id);
    if (graph_iter == func_graph_dynamic_infos_.end()) {
      MS_LOG(EXCEPTION) << "cell id[" << func_graph_cell_id << "] is not found in func_graph_dynamic_infos_";
    }

    auto tensor_list = graph_iter->second.value_node_tensor_list;
    for (auto tensor_info : tensor_list) {
      MS_EXCEPTION_IF_NULL(tensor_info.second);
      tensor_info.second->set_device_address(nullptr);
    }
  }
  MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
}

void UpdateTensorInfo(const tensor::TensorPtr &new_tensor, const tensor::TensorPtr &pre_tensor) {
  MS_EXCEPTION_IF_NULL(new_tensor);
  MS_EXCEPTION_IF_NULL(pre_tensor);
  if (new_tensor->device_address() == nullptr) {
    MS_LOG(DEBUG) << "The number of pre tensors is zero or the device address of new tensor is nullptr.";
    return;
  }
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  MS_EXCEPTION_IF_NULL(pre_tensor);
  MS_LOG(DEBUG) << "Replace Old tensor id " << pre_tensor->id() << " device_address: " << pre_tensor->device_address()
                << " shape and type " << pre_tensor->GetShapeAndDataTypeInfo() << " with New tensor id "
                << new_tensor->id() << " device_address " << new_tensor->device_address() << " shape and dtype "
                << new_tensor->GetShapeAndDataTypeInfo();
  (void)pre_tensor->set_shape(new_tensor->shape());
  (void)pre_tensor->set_data_type(new_tensor->data_type());
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_target != kCPUDevice && device_address->GetDeviceType() != device::DeviceType::kCPU) {
    pre_tensor->set_device_address(new_tensor->device_address());
    return;
  }

  // Replace data in device address when run in CPU device.
  if (pre_tensor->device_address() != nullptr) {
    auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(pre_tensor->device_address());
    MS_EXCEPTION_IF_NULL(old_device_address);
    auto new_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
    MS_EXCEPTION_IF_NULL(new_device_address);

    // CPU host tensor data_c is different from device address if the address is from mem_pool.
    if (new_device_address->from_mem_pool()) {
      pre_tensor->set_device_address(new_device_address);
      return;
    }

    auto old_ptr = old_device_address->GetMutablePtr();
    MS_EXCEPTION_IF_NULL(old_ptr);
    auto new_ptr = new_device_address->GetPtr();
    MS_EXCEPTION_IF_NULL(new_ptr);
    MS_EXCEPTION_IF_CHECK_FAIL(old_device_address->GetSize() == new_device_address->GetSize(), "Size not equal");
    if (old_device_address->GetSize() < SECUREC_MEM_MAX_LEN) {
      auto ret_code = memcpy_s(old_ptr, old_device_address->GetSize(), new_ptr, new_device_address->GetSize());
      MS_EXCEPTION_IF_CHECK_FAIL(ret_code == EOK, "Memory copy failed, ret code: " + std::to_string(ret_code));
    } else {
      auto ret_code = std::memcpy(old_ptr, new_ptr, old_device_address->GetSize());
      MS_EXCEPTION_IF_CHECK_FAIL(ret_code == old_ptr, "Memory copy failed");
    }
  } else {
    pre_tensor->set_device_address(device_address);
    pre_tensor->data_sync();
    pre_tensor->set_device_address(nullptr);
    pre_tensor->set_sync_status(kNeedSyncHostToDevice);
  }
}

void GetValueNodeTensorInfo(const CNodePtr &cnode, const size_t &cur_node_idx,
                            std::vector<std::pair<std::string, tensor::TensorPtr>> *value_node_tensor_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(value_node_tensor_list);
  const auto &inputs = cnode->inputs();
  for (size_t idx = 1; idx < inputs.size(); idx++) {
    if (!inputs[idx]->isa<ValueNode>()) {
      continue;
    }

    auto value = GetValueNode(inputs[idx]);
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<tensor::Tensor>()) {
      continue;
    }

    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    std::string tensor_idx = std::to_string(cur_node_idx) + "_" + std::to_string(idx);
    (void)value_node_tensor_list->emplace_back(std::make_pair(tensor_idx, tensor));
  }
}

bool TopoCompareFuncGraphNode(const AnfNodePtr &node, const bool &is_first_sort,
                              const FuncGraphDynamicInfo *old_graph_info, FuncGraphDynamicInfo *new_graph_info,
                              const size_t &cur_node_idx, const size_t &topo_node_idx) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_graph_info);
  if (!is_first_sort) {
    MS_EXCEPTION_IF_NULL(old_graph_info);
  }
  compile::TopoSortNodeInfo node_info;

  // Parse func graph node.
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsValueNode<Primitive>(cnode->input(0))) {
    node_info.prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    node_info.abs = node->abstract();
  } else if (IsValueNode<FuncGraph>(cnode->input(0))) {
    node_info.is_func_graph = true;
    node_info.func_graph_cell_id = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrFuncGraphCellId);
  } else {
    MS_LOG(EXCEPTION) << "node is not primitive or func graph, cnode->input(0):" << cnode->input(0)->ToString();
  }

  // Compare node info.
  auto compare_func_graph_node = [](const compile::TopoSortNodeInfo &new_node_info,
                                    const compile::TopoSortNodeInfo &old_node_info) {
    if (new_node_info.is_func_graph != old_node_info.is_func_graph ||
        (new_node_info.is_func_graph && new_node_info.func_graph_cell_id != old_node_info.func_graph_cell_id)) {
      return true;
    }

    if (!common::IsEqual(new_node_info.prim, old_node_info.prim)) {
      return true;
    }

    if (new_node_info.abs == nullptr || old_node_info.abs == nullptr) {
      return new_node_info.abs != old_node_info.abs;
    }

    if (!common::IsEqual(new_node_info.abs->BuildType(), old_node_info.abs->BuildType()) ||
        !common::IsEqual(new_node_info.abs->BuildShape(), old_node_info.abs->BuildShape())) {
      return true;
    }
    return false;
  };

  GetValueNodeTensorInfo(cnode, cur_node_idx, &new_graph_info->value_node_tensor_list);
  if (is_first_sort) {
    (void)new_graph_info->topo_node_list.emplace_back(std::move(node_info));
  } else {
    if (common::AnfAlgo::IsControlOpExecInBackend(cnode)) {
      return true;
    }
    if (topo_node_idx >= old_graph_info->topo_node_list.size()) {
      return true;
    }

    if (compare_func_graph_node(node_info, old_graph_info->topo_node_list[topo_node_idx])) {
      return true;
    }
  }
  return false;
}

bool TopoCompareFuncGraphEdge(const bool &is_first_sort, const FuncGraphDynamicInfo *old_graph_info,
                              FuncGraphDynamicInfo *new_graph_info, const size_t &cur_idx, const size_t &new_idx,
                              const size_t &graph_edge_idx) {
  MS_EXCEPTION_IF_NULL(new_graph_info);
  if (!is_first_sort) {
    MS_EXCEPTION_IF_NULL(old_graph_info);
  }

  if (is_first_sort) {
    (void)new_graph_info->graph_edge_list.emplace_back(std::make_pair(cur_idx, new_idx));
  } else {
    if (graph_edge_idx >= old_graph_info->graph_edge_list.size()) {
      return true;
    }

    const auto &graph_edge_info = old_graph_info->graph_edge_list[graph_edge_idx];

    if (cur_idx != graph_edge_info.first || new_idx != graph_edge_info.second) {
      return true;
    }
  }
  return false;
}

bool MindRTBackend::TopoSortGraphCompare(const AnfNodePtr &root, const SuccFunc &succ,
                                         FuncGraphDynamicInfo *new_graph_info, bool is_first_sort,
                                         const FuncGraphDynamicInfo *old_graph_info) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(new_graph_info);
  if (root == nullptr) {
    if (old_graph_info == nullptr) {
      return true;
    }
    return !old_graph_info->topo_node_list.empty();
  }

  size_t topo_node_idx = 0;
  size_t graph_edge_idx = 0;
  if (!is_first_sort) {
    MS_EXCEPTION_IF_NULL(old_graph_info);
  }

  size_t new_idx = 0;
  auto seen = NewSeenGeneration();
  std::stack<AnfNodePtr> todo_nodes;
  (void)todo_nodes.push(root);

  std::unordered_map<size_t, size_t> hash_idx_map;
  hash_idx_map[root->hash()] = new_idx;

  while (!todo_nodes.empty()) {
    AnfNodePtr &node = todo_nodes.top();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_idx = hash_idx_map[node->hash()];
    if (node->extra_seen_ == seen) {
      todo_nodes.pop();
      continue;
    }
    if (node->seen_ == seen) {
      node->extra_seen_ = seen;

      if (TopoCompareFuncGraphNode(node, is_first_sort, old_graph_info, new_graph_info, cur_idx, topo_node_idx)) {
        return true;
      }
      topo_node_idx++;
      todo_nodes.pop();
      continue;
    }
    node->seen_ = seen;
    for (auto &next : succ(node)) {
      if (next == nullptr || !next->isa<CNode>() || next->extra_seen_ == seen) {
        continue;
      }
      if (next->seen_ != seen) {
        new_idx++;
        hash_idx_map[next->hash()] = new_idx;
        if (TopoCompareFuncGraphEdge(is_first_sort, old_graph_info, new_graph_info, cur_idx, new_idx, graph_edge_idx)) {
          return true;
        }
        graph_edge_idx++;
        (void)todo_nodes.push(std::move(next));
        continue;
      }
      auto fg = next->func_graph();
      if (fg != nullptr && fg->return_node() == next) {
        continue;
      }
      MS_LOG(EXCEPTION) << "Graph cycle exists, strike node: " << next->DebugString();
    }
  }
  return false;
}

bool MindRTBackend::IsFuncGraphDynamicShapeOrStruct(const FuncGraphPtr &func_graph, const std::string &cell_id) {
  auto func_graph_iter = func_graph_dynamic_infos_.find(cell_id);
  bool is_first_sort = true;
  FuncGraphDynamicInfo *old_graph_info = nullptr;
  if (func_graph_iter != func_graph_dynamic_infos_.end()) {
    is_first_sort = false;
    old_graph_info = &(func_graph_iter->second);
  }

  if (!is_first_sort && old_graph_info->is_dynamic) {
    MS_LOG(DEBUG) << "graph is dynamic, graph:" << func_graph->ToString();
    return true;
  }

  FuncGraphDynamicInfo new_func_graph_info;
  auto is_graph_dynamic = TopoSortGraphCompare(func_graph->get_return(), SuccDeeperSimple, &new_func_graph_info,
                                               is_first_sort, old_graph_info);
  MS_LOG(DEBUG) << "TopoSortGraphCompare, graph:" << func_graph->ToString() << ", is first sort:" << is_first_sort
                << ", is dynamic:" << is_graph_dynamic;
  if (is_first_sort) {
    func_graph_dynamic_infos_[cell_id] = new_func_graph_info;
    return false;
  } else {
    old_graph_info->is_dynamic = is_graph_dynamic;
    if (old_graph_info->value_node_tensor_list.size() != new_func_graph_info.value_node_tensor_list.size()) {
      MS_LOG(DEBUG) << "graph is dynamic, old graph tensor size:" << old_graph_info->value_node_tensor_list.size()
                    << ", new graph tensor size:" << new_func_graph_info.value_node_tensor_list.size();
      old_graph_info->is_dynamic = true;
      return true;
    }

    for (size_t i = 0; i < new_func_graph_info.value_node_tensor_list.size(); ++i) {
      const auto &old_tensor_info = old_graph_info->value_node_tensor_list[i];
      const auto &new_tensor_info = new_func_graph_info.value_node_tensor_list[i];

      if (old_tensor_info.first != new_tensor_info.first) {
        MS_LOG(DEBUG) << "graph is dynamic, tensor idx:" << i << ", old tensor info:" << old_tensor_info.first
                      << ", new tensor info:" << new_tensor_info.first;
        old_graph_info->is_dynamic = true;
        return true;
      }

      UpdateTensorInfo(new_tensor_info.second, old_tensor_info.second);
    }
  }

  return is_graph_dynamic;
}

void MindRTBackend::WaitTaskFinish() const { runtime::OpExecutor::GetInstance().Wait(); }

void MindRTBackend::ClearOpExecutorResource() const { runtime::OpExecutor::GetInstance().Reset(); }

void MindRTBackend::SyncStream() {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }
}

void MindRTBackend::EraseSingleOpCache(const GraphInfo &graph_info) const {
  pynative::OpCompiler::GetInstance().ClearOpCache(graph_info);
}

void MindRTBackend::ReleaseForwardOutput(const std::vector<TensorPtr> &input_tensors) {
  graph_compiler_->UpdateForwardOpOutputRefCount(input_tensors, &forward_op_output_tensor_id_);
}

void MindRTBackend::CompileSingleOpGraphs(
  const std::vector<std::shared_ptr<pynative::BackendOpBuildTask>> &build_tasks) {
  if (build_tasks.empty()) {
    return;
  }
  std::vector<KernelGraphPtr> graphs;
  for (const auto &task : build_tasks) {
    MS_EXCEPTION_IF_NULL(task);
    const auto &context = task->context();
    MS_EXCEPTION_IF_NULL(context);
    graphs.push_back(context->graph());
  }
  MS_EXCEPTION_IF_NULL(build_tasks[0]);
  auto &task_context = build_tasks[0]->context();
  MS_EXCEPTION_IF_NULL(task_context);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, task_context->is_pynative_infer());

  auto device_context = task_context->device_context();
  pynative::OpCompiler::BatchBuild(graphs, device_context);
}

void MindRTBackend::OpRunCallback(const std::shared_ptr<pynative::OpTaskContext> &context) {
  MS_LOG(DEBUG) << "OpRunCallback start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());
  bool use_dynamic_shape_process = context->op_run_info()->base_op_run_info.use_dynamic_shape_process;
  if (use_dynamic_shape_process) {
    runtime::RunSingleOpGraphDynamic(context->graph(), GetTensorWithoutValueMask(context->op_run_info()),
                                     context->device_context(), context->op_run_info()->is_gradient_out);
  } else {
    runtime::RunSingleOpGraph(context->graph(), GetTensorWithoutValueMask(context->op_run_info()),
                              context->device_context());
  }

  if (!context->op_run_info()->is_infer) {
    ReleaseForwardOutput(context->op_run_info()->base_op_run_info.input_tensor);
  }
  if (use_dynamic_shape_process) {
    ClearGraphDeviceAddressDynamic(context->graph(), context->device_context(),
                                   context->op_run_info()->is_gradient_out);
    ClearInputDeviceAddressDynamic(context->graph(), context->device_context());
  } else {
    ClearGraphDeviceAddress(context->graph(), context->device_context(), context->op_run_info()->is_gradient_out);
    ClearInputDeviceAddress(context->graph(), context->device_context());
  }

  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

void MindRTBackend::BatchBuildCallback() {
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (op_executor.BuildQueueEmpty()) {
    return;
  }

  try {
    MS_LOG(DEBUG) << "Start";
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);

    CompileSingleOpGraphs(op_executor.GetOpBuildTasks());
    op_executor.ClearOpBuildTasks();

    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
    MS_LOG(DEBUG) << "End";
  } catch (const py::type_error &ex) {
    op_executor.Reset();
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    op_executor.Reset();
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    op_executor.Reset();
    throw py::index_error(ex);
  } catch (const py::name_error &ex) {
    op_executor.Reset();
    throw py::name_error(ex);
  } catch (const std::exception &ex) {
    op_executor.Reset();
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    op_executor.Reset();
#ifndef _MSC_VER
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when execute task in queue. Exception name: " << exName;
#else
    MS_LOG(EXCEPTION) << "Error occurred when execute task in queue.";
#endif
  }
}

void MindRTBackend::DispatchOpTask(bool single_op_cache_hit, VectorRef *outputs,
                                   const OpCompilerInfoPtr &op_compiler_info,
                                   const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;

  runtime::UpdateDeviceAddress(graph, GetTensorWithoutValueMask(op_run_info), op_compiler_info->device_context_);
  // Create output tensor
  UpdateOutput(output_nodes, outputs);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  auto run_op_context = std::make_shared<pynative::OpTaskContext>(graph->graph_id(), graph, output_nodes, op_run_info,
                                                                  op_compiler_info->device_context_, infer_flag);

  // Save build task and run task.
  std::promise<bool> promise;
  auto future = promise.get_future();

  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!single_op_cache_hit) {
    op_executor.PushOpBuildTask(std::make_shared<pynative::BackendOpBuildTask>(run_op_context, std::move(promise)));
  } else {
    promise.set_value(true);
  }
  op_executor.PushOpRunTask(std::make_shared<pynative::BackendOpRunTask>(
    run_op_context, [this](const std::shared_ptr<pynative::OpTaskContext> &ctx) { OpRunCallback(ctx); },
    std::move(future)));

  op_executor.Register([this]() { BatchBuildCallback(); });
  if (op_executor.BuildQueueFull()) {
    WaitTaskFinish();
  }
}

void MindRTBackend::RunOpImpl(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                              const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  // Fetch outputs.
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = op_compiler_info->device_context_;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  bool is_dynamic_shape = op_run_info->base_op_run_info.has_dynamic_output;
  bool async_exec_disabled = is_dynamic_shape || op_compiler_info->need_erase_ ||
                             !op_run_info->base_op_run_info.lazy_build || OpInBlackList(op_run_info) ||
                             GetExecutionMode() == kGraphMode || EnablePyNativeSyncRunning();
  if (!async_exec_disabled) {
    MS_LOG(DEBUG) << "Async exec enabled, op: " << op_run_info->base_op_run_info.op_name;
    DispatchOpTask(single_op_cache_hit, outputs, op_compiler_info, op_run_info);
    return;
  }

  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  if (!op_executor.RunQueueEmpty()) {
    WaitTaskFinish();
  }
  if (!single_op_cache_hit) {
    CompileSingleOpGraph(graph, device_context);
  }
  const auto &tensors_without_value_mask = GetTensorWithoutValueMask(op_run_info);
  runtime::UpdateDeviceAddress(graph, tensors_without_value_mask, device_context);

  runtime::RunSingleOpGraph(graph, tensors_without_value_mask, device_context);

  if (!op_run_info->is_infer) {
    ReleaseForwardOutput(op_run_info->base_op_run_info.input_tensor);
  }
  UpdateOutput(output_nodes, outputs);

  ClearGraphDeviceAddress(graph, device_context, op_run_info->is_gradient_out);
  ClearInputDeviceAddress(graph, device_context);

  if (is_dynamic_shape) {
    UpdateOutputAbstract(graph, op_run_info);
  }
  if (op_compiler_info->need_erase_) {
    EraseSingleOpCache(op_run_info->base_op_run_info.graph_info);
  }
}

void MindRTBackend::RunOpImplDynamic(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                                     const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  // Fetch outputs.
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = op_compiler_info->device_context_;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  bool is_dynamic_shape = op_run_info->base_op_run_info.has_dynamic_output;
  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  if (!op_executor.RunQueueEmpty()) {
    WaitTaskFinish();
  }
  if (!single_op_cache_hit) {
    CompileSingleOpGraph(graph, device_context);
  }
  const auto &tensors_without_value_mask = GetTensorWithoutValueMask(op_run_info);
  runtime::UpdateDeviceAddress(graph, tensors_without_value_mask, device_context);
  runtime::RunSingleOpGraphDynamic(graph, tensors_without_value_mask, device_context, op_run_info->is_gradient_out);

  if (!op_run_info->is_infer) {
    ReleaseForwardOutput(op_run_info->base_op_run_info.input_tensor);
  }
  UpdateOutput(output_nodes, outputs);
  ClearGraphDeviceAddressDynamic(graph, device_context, op_run_info->is_gradient_out);
  ClearInputDeviceAddressDynamic(graph, device_context);
  if (is_dynamic_shape) {
    UpdateOutputAbstract(graph, op_run_info);
  }
  if (op_compiler_info->need_erase_) {
    EraseSingleOpCache(op_run_info->base_op_run_info.graph_info);
  }
}

void MindRTBackend::RunOp(const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_LOG(INFO) << "RunOp start " << op_run_info->base_op_run_info.op_name;
  // Get the device context.
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_context);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  if (runtime::OpExecutor::GetInstance().ActorInQueue(op_compiler_info->graph_id_)) {
    WaitTaskFinish();
  }

  if (!single_op_cache_hit) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool enable_cache = context_ptr->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
    // If op not support dynamic shape, op will select static opinfo, update graph dynamic attr
    op_compiler_info->need_erase_ = !enable_cache;
  }

  RunOpImpl(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void MindRTBackend::RunOpDynamic(const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_LOG(INFO) << "RunOp start " << op_run_info->base_op_run_info.op_name;
  // Get the device context.
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_context);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  if (runtime::OpExecutor::GetInstance().ActorInQueue(op_compiler_info->graph_id_)) {
    WaitTaskFinish();
  }

  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);

  if (!single_op_cache_hit) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool enable_cache = context_ptr->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
    op_compiler_info->need_erase_ = !enable_cache;
  } else {
    auto input_nodes = graph->input_nodes();
    auto input_size = input_nodes.size();
    auto input_tensors = GetTensorWithoutValueMask(op_run_info);
    if (input_size > input_tensors.size()) {
      MS_LOG(EXCEPTION) << "input_size is bigger than input_tensors size, input_size:" << input_size
                        << ", input_tensors size:" << input_tensors.size();
    }
    // Update the Graph`s Parameter shape
    for (size_t i = 0; i < input_size; ++i) {
      MS_EXCEPTION_IF_NULL(input_tensors[i]);
      auto type_of_tensor = input_tensors[i]->Dtype();
      std::shared_ptr<abstract::AbstractTensor> abstract;
      abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, input_tensors[i]->shape());
      input_nodes[i]->set_abstract(abstract);
    }

    // Create input address before infer
    runtime::DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);
    runtime::DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  }

  RunOpImplDynamic(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void MindRTBackend::CompileSingleOpGraph(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  pynative::OpCompiler::BatchBuild({graph}, device_context);
}

void MindRTBackend::UpdateOutput(const std::vector<session::KernelWithIndex> &output_nodes,
                                 VectorRef *const outputs) const {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (common::AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto output_tensor = CreateOutputTensor(item_with_index.first, item_with_index.second);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().Wait(); });
    outputs->emplace_back(output_tensor);
  }
}

void MindRTBackend::ClearResource() {
  graph_compiler_ = std::make_shared<GraphCompiler>();
  graph_id_to_device_context_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  graph_info_to_device_context_.clear();
  control_nodes_.clear();
  actor_to_graph_compiler_info_.clear();
  cnode_ref_counts_.clear();
}

KernelGraphPtr MindRTBackend::GetGraphById(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  return graph_compiler_->Fetch(graph_id);
}
}  // namespace compile
}  // namespace mindspore
