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
#include "ops/sequence_ops.h"
#include "ops/nn_op_name.h"
#include "ops/structure_op_name.h"
#include "include/common/utils/parallel_context.h"
#include "backend/graph_compiler/transform.h"
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "include/backend/optimizer/helper.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/pynative/grad/jit/jit_call_graph.h"
#include "ir/anf.h"
#include "pybind_api/ir/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/callbacks.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/pynative/run_op_helper.h"
#include "runtime/pynative/graph_adapter.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "pybind_api/gil_scoped_long_running.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
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
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
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
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetWorkspaceAddr(new_device_address, i, node.get());
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
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
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

int GetExecutionMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
}

void UpdateTensorAddress(const session::BackendOpRunInfoPtr &op_run_info,
                         const std::vector<session::KernelWithIndex> &output_nodes, VectorRef *const outputs) {
  auto &output_tensors = op_run_info->base_op_run_info.output_tensors;
  auto &output_promises = op_run_info->device_sync_promises;
  if (output_tensors.size() != output_nodes.size() || output_tensors.size() != output_promises.size()) {
    MS_LOG(EXCEPTION) << "Output tensors " << output_tensors.size() << " output promises " << output_promises.size()
                      << " output nodes " << output_nodes.size();
  }

  auto exec_mode = GetExecutionMode();
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto &node_index = output_nodes[i];
    if (AnfAlgo::GetOutputTensorNum(node_index.first) == 0) {
      continue;
    }
    auto &output_tensor = output_tensors[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    auto &output_promise = output_promises[i];
    MS_EXCEPTION_IF_NULL(output_promise);

    const auto &device_address = AnfAlgo::GetMutableOutputAddr(node_index.first, node_index.second, false);
    device_address->set_padding_type(AnfAlgo::GetOutputReshapeType(node_index.first, node_index.second));
    output_promise->SetValue(std::make_shared<pynative::DeviceAddressFutureData>(device_address, nullptr));
    (void)outputs->emplace_back(output_tensor);

    if (exec_mode != kPynativeMode) {
      output_tensor->data_sync(false);
    }
  }
}

void UpdateTensorAddressDynamic(const session::BackendOpRunInfoPtr &op_run_info,
                                const OpCompilerInfoPtr &op_compiler_info,
                                const vector<device::DeviceAddressPtr> &device_address_list, VectorRef *const outputs) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto &output_tensors = op_run_info->base_op_run_info.output_tensors;
  auto &output_promises = op_run_info->device_sync_promises;
  auto &output_nodes = op_compiler_info->graph_output_nodes_;
  if (output_tensors.size() != output_nodes.size() || output_tensors.size() != output_promises.size()) {
    MS_LOG(EXCEPTION) << "Output tensors " << output_tensors.size() << " output promises " << output_promises.size()
                      << " output nodes " << output_nodes.size();
  }

  auto exec_mode = GetExecutionMode();
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    if (op_compiler_info->graph_outputs_tensor_num_[i] == 0) {
      continue;
    }
    auto &output_tensor = output_tensors[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    auto &output_promise = output_promises[i];
    MS_EXCEPTION_IF_NULL(output_promise);
    device_address_list[i]->set_padding_type(op_compiler_info->graph_outputs_padding_type_[i]);
    output_promise->SetValue(std::make_shared<pynative::DeviceAddressFutureData>(device_address_list[i], nullptr));
    (void)outputs->emplace_back(output_tensor);
    if (exec_mode != kPynativeMode) {
      output_tensor->data_sync(false);
    }
  }
}

void AllocateMemForTensor(const tensor::TensorPtr &tensor, DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  if ((device_address->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(device_address.get()))) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }

  auto tensor_size = LongToSize(tensor->data().nbytes());
  auto tensor_type = tensor->data_type();
  if (!device_address->SyncHostToDevice(tensor->shape(), tensor_size, tensor_type, tensor->data_c(),
                                        tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
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

GraphId MsBackend::CompileGraph(const NotNull<FuncGraphPtr> &fg) {
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
ValuePtr GetInputofBpropCut(const std::shared_ptr<GraphCompiler> &graph_compiler, const CNodePtr &parent_node,
                            const AnfNodePtr &input_node, const std::map<KernelWithIndex, TensorPtr> &op_output,
                            const std::map<AnfNodePtr, size_t> &parameter_index,
                            const std::vector<TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                            size_t input_index) {
  if (!IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
    auto real_input = common::AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(real_input);
    ValuePtr value = nullptr;
    if (!real_input->isa<ValueNode>()) {
      if (real_input->abstract() != nullptr && real_input->abstract()->isa<abstract::AbstractSparseTensor>()) {
        value = TensorListToSparseTensor(real_input->abstract(), graph_inputs);
      } else {
        value = graph_compiler->GetSingleOpInputTensorByIndex(parent_node, op_output, parameter_index, graph_inputs,
                                                              input_tensor_info, input_index);
      }
      MS_EXCEPTION_IF_NULL(value);
    } else {
      const auto &value_node = real_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
    }
    return value;
  }
  auto cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::vector<ValuePtr> args_tuple;
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->inputs()[i];
    auto value = GetInputofBpropCut(graph_compiler, cnode, input, op_output, parameter_index, graph_inputs,
                                    input_tensor_info, i - 1);
    MS_EXCEPTION_IF_NULL(value);
    (void)args_tuple.emplace_back(value);
  }
  auto arg = std::make_shared<ValueTuple>(args_tuple);
  return arg;
}

ValuePtr GetFrontArgByParameter(const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                                const AnfNodePtr &front_node) {
  const auto &iter = std::find(origin_paramters.begin(), origin_paramters.end(), front_node);
  const size_t index = iter - origin_paramters.begin();
  // If the parameter is not found in the parameters of the root graph, it means that it is the input of the subgraph,
  // and there is no need to input a tensor.
  if (index >= front_args.size()) {
    MS_LOG(EXCEPTION) << "Position out of front args range, position value is " << index << " and args size is "
                      << front_args.size() << ".";
  }
  auto value = utils::cast<ValuePtr>(front_args[index]);
  MS_EXCEPTION_IF_NULL(value);
  return value;
}

void GetControlOpInput(const std::shared_ptr<GraphCompiler> &graph_compiler,
                       const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                       const CNodePtr &front_cnode, const CNodePtr &backend_cnode,
                       const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                       const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                       VectorRef *args) {
  MS_EXCEPTION_IF_NULL(front_cnode);
  MS_EXCEPTION_IF_NULL(backend_cnode);
  MS_EXCEPTION_IF_NULL(graph_compiler);
  MS_EXCEPTION_IF_NULL(args);
  auto front_size = front_cnode->inputs().size();
  auto back_size = backend_cnode->inputs().size();
  if (front_size != back_size) {
    MS_LOG(EXCEPTION) << "Bpropcut op front cnode size: " << front_size << ", back cnode size:" << back_size
                      << ", bpropcut op should not flatten";
  }
  for (size_t index = 1; index < back_size; ++index) {
    auto input_node = backend_cnode->input(index);
    ValuePtr value = nullptr;
    if (input_node->isa<Parameter>() && input_node->abstract() != nullptr &&
        input_node->abstract()->isa<abstract::AbstractSequence>()) {
      auto front_input_node = front_cnode->input(index);
      value = GetFrontArgByParameter(origin_paramters, front_args, front_input_node);
    } else {
      value = GetInputofBpropCut(graph_compiler, backend_cnode, input_node, op_output_map, parameter_index,
                                 graph_inputs, input_tensor_info, index - 1);
    }
    MS_EXCEPTION_IF_NULL(value);
    (void)args->emplace_back(value);
  }
}

void ConvertPyObjectToTensor(const py::object &input_object, std::vector<ValuePtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  ValuePtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (IsStubTensor(input_object)) {
    tensor_ptr = ConvertStubTensor(input_object);
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

void RunControlOperator(const std::shared_ptr<GraphCompiler> &graph_compiler,
                        const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                        const KernelGraphPtr &graph, const CNodePtr &kernel,
                        const std::map<KernelWithIndex, tensor::TensorPtr> &op_output_map,
                        const std::map<AnfNodePtr, size_t> &parameter_index,
                        const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info,
                        VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_outputs);
  AnfNodePtr front_node = graph->GetFrontAnfByBackendAnf(kernel);
  if (front_node == nullptr && graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    front_node = kernel;
  }
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
    GetControlOpInput(graph_compiler, origin_paramters, front_args, cnode, kernel, op_output_map, parameter_index,
                      graph_inputs, input_tensor_info, &args);
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

void UpdateOutputAbstract(const VectorRef &outputs, const session::BackendOpRunInfoPtr &op_run_info) {
  auto output_size = outputs.size();
  if (output_size == 1 && op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    auto output_tensor = utils::cast<tensor::TensorPtr>(outputs[0]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    op_run_info->base_op_run_info.abstract = output_tensor->ToAbstract();
    MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                  << op_run_info->base_op_run_info.abstract->ToString();
    return;
  }
  AbstractBasePtrList elements;
  for (size_t i = 0; i < output_size; ++i) {
    auto output_tensor = utils::cast<tensor::TensorPtr>(outputs[i]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    (void)elements.emplace_back(output_tensor->ToAbstract());
  }
  op_run_info->base_op_run_info.abstract = std::make_shared<abstract::AbstractTuple>(elements);
  MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                << op_run_info->base_op_run_info.abstract->ToString();
}

TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &abstract = common::AnfAlgo::GetNodeAbstractByIndex(output_node, output_index);
  if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
    return AnfAlgo::CreateMapTensor(output_node, output_index);
  }
  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
  const auto &shape = common::AnfAlgo::GetOutputInferShape(output_node, output_index);
  auto tensor = std::make_shared<tensor::Tensor>(type_id, shape);

  // Put device tensor into host tensor.
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->SetNodeIndex(output_node, output_index);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));
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
TensorPtr CreateOutputTensorDynamicImpl(const OpCompilerInfoPtr &op_compiler_info, const AnfNodePtr &output_node,
                                        size_t output_index, const std::shared_ptr<device::DeviceAddress> &address,
                                        size_t idx_in_graph_outputs) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(op_compiler_info);

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto tensor = std::make_shared<tensor::Tensor>(address->type_id(), address->host_shape());

  // Put device tensor into host tensor.
  address->SetNodeIndex(output_node, output_index);
  address->set_padding_type(op_compiler_info->graph_outputs_padding_type_[idx_in_graph_outputs]);
  tensor->set_device_address(address);

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

#if !defined(__APPLE__)
bool EnablePyNativeSyncRunning() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
}
#endif

bool DisableRunOpAsync(const OpCompilerInfoPtr &op_compiler_info, const session::BackendOpRunInfoPtr &op_run_info) {
#if defined(__APPLE__)
  return true;
#else
  return op_run_info->base_op_run_info.has_dynamic_output ||  // Infer output is dynamic.
         op_compiler_info->need_refresh_abstract_ ||          // Graph output is dynamic after IR Pass. (e.g. Dropout)
         op_compiler_info->need_erase_ ||                     // Random op cache need to be erased.
         GetExecutionMode() == kGraphMode ||                  // Cannot find a wait point before compile graph.
         EnablePyNativeSyncRunning();                         // context.set_context(pynative_synchronize=True)
#endif
}
}  // namespace

runtime::ActorSet *MindRTBackend::RealCompileGraphBeforeRunActor(const GraphCompilerInfo &graph_compiler_info,
                                                                 const VectorRef &args, bool no_multi_graph) {
  auto graphs = graph_compiler_info.graphs_;
  auto device_contexts = graph_compiler_info.device_contexts_;

  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_flag(kFlagPyNativeRunInGraph, true);
    graph->set_flag(kFlagIsPynativeBpropGraph, root_graph_->has_flag(kFlagIsPynativeBpropGraph));
    if (graph->is_any_type_input()) {
      continue;
    }
    if (no_multi_graph) {
      MS_LOG(INFO) << "Replace parameter format";
      // The input tensors of heterogeneous graphs or control flow graphs are null.
      // Need to get tensor after ParseControlNodes.
      auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
      pynative::GraphAdapter::ReplaceGraphParameterProperties(graph, input_tensors.at(i), device_contexts[i]);
    }
    (void)graph_compiler_->CompileGraphImpl(graph, device_contexts[i]);
    pynative::GraphAdapter::RemoveUnusedValueNodes(graph);
    // PyNative use kernel graph will result in front node and back node is the same; But in pynative task sink, backend
    // still create new kernel graph
    if (root_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph) &&
        !pynative::GraphAdapter::PyNativeEnableTaskSink(root_graph_)) {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, {graph->output()});
    } else {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, graph->front_outputs());
    }
    // Clear front outputs after the outputs is cached.
    graph->set_front_outputs({});
    AnfAlgo::UpdateGraphValidRefPair(graph);
    pynative::GraphAdapter::SensTensorToDevice(graph, device_contexts[i]);
  }

  ParseControlNodes(graph_compiler_info);
  UpdateGraphCompilerInfo(graph_compiler_info.name_);
  auto actor_set = runtime::GraphScheduler::GetInstance().Transform(graph_compiler_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  constexpr auto kKernelActorThreshold = 5000;
  // Turning off multithreading may cause stack overflow in control flow scenarios.
  if (no_multi_graph && actor_set->kernel_actors_.size() < kKernelActorThreshold &&
      root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    // Multithreading can cause spikes in memory usage and performance fluctuations.
    actor_set->is_multi_thread_execution_ = false;
    MS_LOG(INFO) << "Actor Multithreading is turned off!";
  }
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);

  for (auto &graph : graphs) {
    pynative::GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(graph);
    pynative::GraphAdapter::GenerateRefCountForBpropValueNode(graph);
  }
  return actor_set;
}

void MindRTBackend::RunGraphByActors(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                                     const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: begin run actor: " << actor_info;
  WaitTaskFinish();
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto graphs = graph_compiler_info.graphs_;
  auto device_contexts = graph_compiler_info.device_contexts_;
  if (device_contexts.size() != graphs.size()) {
    MS_LOG(EXCEPTION) << "Graphs size " << graphs.size() << " is not equal to device_contexts size "
                      << device_contexts.size();
  }

  // KernelByKernel: The size of control_nodes is at least 1 since there is return node in the graph.
  // GraphMode: No control nodes.
  bool no_multi_graph = control_nodes_.size() <= 1 && graphs.size() == 1;
  auto actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  if (actor_set == nullptr) {
    actor_set = RealCompileGraphBeforeRunActor(graph_compiler_info, args, no_multi_graph);
  }

  if (root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    for (size_t i = 0; i < graphs.size(); ++i) {
      pynative::GraphAdapter::UpdateForwardOutputInBpropGraph(graphs[i], device_contexts[i], no_multi_graph);
      pynative::GraphAdapter::UpdateDynamicValueNodeAbstract(graphs[i]);
    }
  }

  auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
  if (graphs.size() > input_tensors.size()) {
    MS_LOG(EXCEPTION) << "The actor_set " << actor_info << " graphs size " << graphs.size()
                      << " should less than or equal to inputs size " << input_tensors.size();
  }
  pynative::GraphAdapter::HandleHeterogeneousTensors(input_tensors, device_contexts);

  // Release GIL and run actor DAG.
  GilReleaseWithCheck release_gil;
  runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors);

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  auto output = root_graph_->output();
  MS_LOG(DEBUG) << "Current out " << output->DebugString();
  if (root_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    MS_EXCEPTION_IF_NULL(output_node_);
    root_graph_->set_output(output_node_);
  }
  ConstructOutputs(actor_set, outputs, root_graph_);
  actor_set->output_actor_->FreeSummaryNodeMem();
  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
}

void MindRTBackend::RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto jit_call_graph = kernel->user_data<pynative::JitCallGraph>();
  MS_EXCEPTION_IF_NULL(jit_call_graph);
  *outputs = jit_call_graph->Run(args);
}

void MindRTBackend::RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                       VectorRef *outputs) {
  WaitTaskFinish();
  auto &op_executor = runtime::OpExecutor::GetInstance();
  op_executor.Register([this]() { BatchBuildCallback(); });

  MS_LOG(INFO) << "Status record: begin run graph by single op";
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
      graph_compiler_->CalculateForwardOpOutputCount(graph, inputs[graph_index], &forward_op_output_tensor_id_,
                                                     parameter_index);
    }

    GilReleaseWithCheck gil_release;
    for (const auto &kernel : graph->execution_order()) {
      MS_LOG(DEBUG) << "Split and run op " << kernel->fullname_with_scope();
      InputTensorInfo input_tensor_info;
      VectorRef op_outputs;
      if (common::AnfAlgo::IsBpropCutOpExecInBackend(kernel)) {
        WaitTaskFinish();
        const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;
        RunControlOperator(graph_compiler_, origin_parameters, args, graph, kernel, op_output_map, parameter_index,
                           inputs[graph_index], &input_tensor_info, &op_outputs);
        // Execute remaining lazy tasks before PyNative hook exit.
        WaitTaskFinish();
      } else if (common::AnfAlgo::HasNodeAttr(kAttrJitCallNode, kernel)) {
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
        auto is_dynamic = common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, kernel);
        session::BackendOpRunInfoPtr op_run_info;
        graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index],
                                                 &input_tensor_info);
        graph_compiler_->GetSingleOpRunInfoAndGraphInfo(kernel, input_tensor_info, is_dynamic, &op_run_info,
                                                        &graph_output_info);
        if (is_dynamic) {
          op_run_info->op_prim = std::make_shared<Primitive>(*op_run_info->op_prim);
          AnfAlgo::SetDynamicAttrToPrim(op_run_info->op_prim);
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
  python_adapter::PyAdapterCallback::ProcessUnPairedCellHook(true);
  MS_LOG(INFO) << "Status record: end run graph by single op";
}

void MindRTBackend::RunGraphByCondition(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                                        const VectorRef &args, VectorRef *outputs) {
  bool enable_run_graph_by_single_op =
    std::any_of(graph_compiler_info.graphs_.begin(), graph_compiler_info.graphs_.end(),
                [](const KernelGraphPtr &graph) { return graph->has_flag(kFlagEnableRunGraphBySingleOp); });
  if (enable_run_graph_by_single_op) {
    RunGraphBySingleOp(graph_compiler_info, args, outputs);
  } else {
    RunGraphByActors(actor_info, graph_compiler_info, args, outputs);
  }
}

void MindRTBackend::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::OpExecutor::GetInstance().WaitAll();
}

void MindRTBackend::ClearOpExecutorResource() const { runtime::OpExecutor::GetInstance().Reset(); }

void MindRTBackend::SyncStream() {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

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
  const std::vector<std::shared_ptr<pynative::DeviceOpBuildTask>> &build_tasks) const {
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
  pynative::OpCompiler::GetInstance().BatchBuild(graphs, device_context);
}

void MindRTBackend::OpRunCallback(const std::shared_ptr<pynative::OpTaskContext> &context) {
  MS_LOG(DEBUG) << "OpRunCallback start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());
  runtime::RunSingleOpGraph(context->graph(), runtime::GetTensorWithoutValueMask(context->op_run_info()),
                            context->device_context());

  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(context->op_run_info());
  if (!context->op_run_info()->is_infer) {
    ReleaseForwardOutput(context->op_run_info()->base_op_run_info.input_tensor);
  }

  ClearGraphDeviceAddress(context->graph(), context->device_context(), context->op_run_info()->is_gradient_out);
  ClearInputDeviceAddress(context->graph(), context->device_context());

  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

void MindRTBackend::OpRunCallbackDynamic(const std::shared_ptr<pynative::OpTaskContext> &context) {
  MS_LOG(DEBUG) << "OpRunCallback start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());

  runtime::RunSingleOpDynamic(context->op_run_info(), context->op_compiler_info(), &context->device_address_list());

  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(context->op_run_info());
  if (!context->op_run_info()->is_infer) {
    ReleaseForwardOutput(context->op_run_info()->base_op_run_info.input_tensor);
  }

  ClearInputDeviceAddressDynamic(context->graph(), context->device_context());
  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

void MindRTBackend::BatchBuildCallback() const {
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (op_executor.BuildQueueEmpty()) {
    return;
  }

  try {
    MS_LOG(DEBUG) << "Start";
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);

    auto build_tasks_in_queue = op_executor.PopOpBuildTasks();
    CompileSingleOpGraphs(build_tasks_in_queue);
    for (auto &task : build_tasks_in_queue) {
      MS_EXCEPTION_IF_NULL(task);
      task->SetBuildReady(true);
    }

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

  runtime::UpdateDeviceAddress(graph, runtime::GetTensorWithoutValueMask(op_run_info),
                               op_compiler_info->device_context_);
  // Create output tensor
  UpdateOutput(op_run_info, op_compiler_info->graph_output_nodes_, outputs);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  auto run_op_context = std::make_shared<pynative::OpTaskContext>(graph->graph_id(), graph, op_run_info,
                                                                  op_compiler_info->device_context_, infer_flag);

  // Save build task and run task.
  std::promise<bool> promise;
  auto future = promise.get_future();

  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!single_op_cache_hit) {
    op_executor.PushOpBuildTask(std::make_shared<pynative::DeviceOpBuildTask>(run_op_context, std::move(promise)));
  } else {
    promise.set_value(true);
  }
  auto run_task = std::make_shared<pynative::DeviceOpRunTask>(
    run_op_context, [this](const std::shared_ptr<pynative::OpTaskContext> &ctx) { OpRunCallback(ctx); },
    std::move(future));
  run_task->set_task_id(op_compiler_info->graph_id_);
  op_executor.PushOpRunTask(run_task);

  op_executor.Register([this]() { BatchBuildCallback(); });
  if (op_executor.BuildQueueFull()) {
    WaitTaskFinish();
  }
}

void MindRTBackend::DispatchOpTaskDynamic(VectorRef *outputs, const OpCompilerInfoPtr &op_compiler_info,
                                          const session::BackendOpRunInfoPtr &op_run_info,
                                          const vector<device::DeviceAddressPtr> &device_address_list) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  auto run_op_context = std::make_shared<pynative::OpTaskContext>(graph->graph_id(), graph, op_run_info,
                                                                  op_compiler_info->device_context_, infer_flag);
  run_op_context->set_op_compiler_info(op_compiler_info);
  run_op_context->set_device_address_list(device_address_list);
  // Save build task and run task.
  std::promise<bool> promise;
  auto future = promise.get_future();

  auto &op_executor = runtime::OpExecutor::GetInstance();
  promise.set_value(true);
  op_executor.PushOpRunTask(std::make_shared<pynative::DeviceOpRunTask>(
    run_op_context, [this](const std::shared_ptr<pynative::OpTaskContext> &ctx) { OpRunCallbackDynamic(ctx); },
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
  if (!DisableRunOpAsync(op_compiler_info, op_run_info)) {
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
  const auto &tensors_without_value_mask = runtime::GetTensorWithoutValueMask(op_run_info);
  runtime::UpdateDeviceAddress(graph, tensors_without_value_mask, device_context);

  runtime::RunSingleOpGraph(graph, tensors_without_value_mask, device_context);

  if (!op_run_info->is_infer) {
    ReleaseForwardOutput(op_run_info->base_op_run_info.input_tensor);
  }
  UpdateOutput(op_run_info, output_nodes, outputs);

  ClearGraphDeviceAddress(graph, device_context, op_run_info->is_gradient_out);
  ClearInputDeviceAddress(graph, device_context);

  if (op_run_info->base_op_run_info.has_dynamic_output || op_compiler_info->need_refresh_abstract_) {
    UpdateOutputAbstract(*outputs, op_run_info);
  }
  if (op_compiler_info->need_erase_) {
    EraseSingleOpCache(op_compiler_info->graph_info_);
  }
}

void GetIgnoreSyncHostToDeviceList(const OpCompilerInfoPtr &op_compiler_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  for (auto const &execute_kernel : op_compiler_info->execute_kernel_list_) {
    auto kernel_mod = AnfAlgo::GetKernelMod(execute_kernel.kernel_);
    std::vector<size_t> ignore_input_index_list;
    if (kernel_mod != nullptr) {
      ignore_input_index_list = kernel_mod->GetLaunchIgnoredInputAddressIdx();
    }
    auto input_device_address_list = execute_kernel.inputs_device_address_;
    for (auto index : ignore_input_index_list) {
      if (index >= input_device_address_list.size()) {
        continue;
      }
      (void)op_compiler_info->ignore_host_to_device_inputs_.emplace(input_device_address_list[index]);
    }
  }
}

void MindRTBackend::RunOpImplDynamic(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                                     const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  MS_LOG(DEBUG) << "RunOpImplDynamic " << op_run_info->base_op_run_info.op_name;
  // Fetch outputs.
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = op_compiler_info->device_context_;
  if (!single_op_cache_hit) {
    CompileSingleOpGraph(op_compiler_info->graph_, device_context, true);
    GetIgnoreSyncHostToDeviceList(op_compiler_info);
  }
  if (!DisableRunOpAsync(op_compiler_info, op_run_info)) {
    MS_LOG(DEBUG) << "Async exec enabled, op: " << op_run_info->base_op_run_info.op_name;
    // Create graph output device address
    auto device_address_list = runtime::DeviceAddressUtils::CreateGraphOutputDeviceAddress(
      op_compiler_info, op_run_info->base_op_run_info.abstract);
    // Create output tensor
    UpdateOutputDynamic(op_run_info, op_compiler_info, device_address_list, outputs);
    DispatchOpTaskDynamic(outputs, op_compiler_info, op_run_info, device_address_list);
    return;
  }
  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!op_executor.RunQueueEmpty()) {
    WaitTaskFinish();
  }

  std::vector<device::DeviceAddressPtr> device_address_list;
  runtime::RunSingleOpDynamic(op_run_info, op_compiler_info, &device_address_list);

  if (!op_run_info->is_infer) {
    ReleaseForwardOutput(op_run_info->base_op_run_info.input_tensor);
  }

  // Create output tensor
  UpdateOutputDynamic(op_run_info, op_compiler_info, device_address_list, outputs);
  ClearInputDeviceAddressDynamic(graph, device_context);
  UpdateOutputAbstract(*outputs, op_run_info);
}

void MindRTBackend::RunOp(const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_LOG(DEBUG) << "Run Op " << op_run_info->base_op_run_info.op_name;

  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_name_, device_id_);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  if (runtime::OpExecutor::GetInstance().ActorInQueue(op_compiler_info->graph_id_)) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                       op_run_info->base_op_run_info.op_name, true);
    runtime::OpExecutor::GetInstance().Wait();
  }

  RunOpImpl(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void MindRTBackend::RunOpDynamic(const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_LOG(DEBUG) << "Run Op " << op_run_info->base_op_run_info.op_name;

  // Single op graph compile
  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_name_, device_id_);
  MS_EXCEPTION_IF_NULL(op_compiler_info);

  RunOpImplDynamic(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void MindRTBackend::RunViewKernelTaskAsyncImpl(const pynative::KernelTaskType &task_type, DeviceContext *device_context,
                                               const device::DeviceAddressPtrList &input_addr_list,
                                               const TensorStorageInfoPtrList &input_storage_list,
                                               const device::DeviceAddressPtrList &output_addr_list) {
  static auto kernel_task_func =
    [](const pynative::KernelTaskType &task_type, const device::DeviceAddressPtrList &input_addr_list,
       const TensorStorageInfoPtrList &input_storage_list, const device::DeviceAddressPtrList &output_addr_list,
       DeviceContext *device_context) {
      runtime::LaunchKernelTask(task_type, device_context, input_addr_list, input_storage_list, output_addr_list);
    };

  auto kernel_task = std::make_shared<pynative::KernelDeviceTask>(
    kernel_task_func, task_type, device_context, input_addr_list, input_storage_list, output_addr_list);
  auto &op_executor = runtime::OpExecutor::GetInstance();
  op_executor.PushSimpleOpRunTask(kernel_task);
}

void MindRTBackend::RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info,
                                      const pynative::KernelTaskType &task_type, bool enable_async) {
  device::DeviceAddressPtrList input_addr_list;
  TensorStorageInfoPtrList input_storage_list;
  device::DeviceAddressPtrList output_addr_list;

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {base_op_run_info.device_target, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);

  for (size_t idx = 0; idx < base_op_run_info.input_tensor.size(); idx++) {
    auto input_tensor = base_op_run_info.input_tensor[idx];
    if (input_tensor->device_address() == nullptr) {
      if (idx == 0) {
        MS_LOG(EXCEPTION) << "First tensor can not be nullptr";
      }
      auto address_size = GetTypeByte(TypeIdToType(input_tensor->data_type())) * SizeOf(input_tensor->shape());
      auto input_addr = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, address_size, kOpFormat_DEFAULT, input_tensor->data_type(), input_tensor->shape());

      input_tensor->set_device_address(input_addr);
      RunAllocMemTask(device_context, input_tensor, enable_async);
      (void)input_addr_list.emplace_back(input_addr);
    } else {
      (void)input_addr_list.emplace_back(
        std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address()));
    }
    (void)input_storage_list.emplace_back(input_tensor->storage_info());
  }

  std::transform(base_op_run_info.output_tensors.begin(), base_op_run_info.output_tensors.end(),
                 std::back_inserter(output_addr_list), [](const auto &tensor) {
                   return std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
                 });

  if (enable_async) {
    RunViewKernelTaskAsyncImpl(task_type, device_context, input_addr_list, input_storage_list, output_addr_list);
  } else {
    WaitTaskFinish();
    runtime::LaunchKernelTask(task_type, device_context, input_addr_list, input_storage_list, output_addr_list);
  }
}

void MindRTBackend::RunContiguousTask(const tensor::TensorPtr &tensor, bool enable_async) {
  MS_EXCEPTION_IF_NULL(tensor);

  auto old_storage_info = tensor->storage_info();
  auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  auto new_device_address = RunContiguousTaskByAddress(old_device_address, old_storage_info, enable_async);
  MS_EXCEPTION_IF_NULL(new_device_address);

  tensor->set_device_address(new_device_address);
  tensor->set_storage_info(nullptr);
}

device::DeviceAddressPtr MindRTBackend::RunContiguousTaskByAddress(const device::DeviceAddressPtr &old_device_address,
                                                                   const TensorStorageInfoPtr &old_storage_info,
                                                                   bool enable_async) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(old_storage_info);

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {old_device_address->device_name(), old_device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);

  auto address_size = GetTypeByte(TypeIdToType(old_device_address->type_id())) * SizeOf(old_storage_info->shape);
  if (old_storage_info->data_type == kTypeUnknown) {
    MS_LOG(EXCEPTION) << "The view op out type is kTypeUnknown";
  }
  auto type_id = old_storage_info->data_type;
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, address_size, kOpFormat_DEFAULT, type_id, old_storage_info->shape);
  new_device_address->set_device_shape(old_storage_info->shape);
  new_device_address->set_original_ref_count(SIZE_MAX);
  new_device_address->ResetRefCount();

  if (enable_async) {
    RunViewKernelTaskAsyncImpl(pynative::KernelTaskType::kCONTIGUOUS_TASK, device_context, {old_device_address},
                               {old_storage_info}, {new_device_address});
  } else {
    WaitTaskFinish();
    runtime::LaunchKernelTask(pynative::KernelTaskType::kCONTIGUOUS_TASK, device_context, {old_device_address},
                              {old_storage_info}, {new_device_address});
  }
  return new_device_address;
}

void MindRTBackend::RunAllocMemTask(DeviceContext *device_context, const tensor::TensorPtr &tensor, bool enable_async) {
  if (!enable_async) {
    WaitTaskFinish();
    return AllocateMemForTensor(tensor, device_context);
  }

  static auto alloc_mem_func = [](DeviceContext *device_context, const tensor::TensorPtr &tensor) {
    AllocateMemForTensor(tensor, device_context);
  };

  auto view_task = std::make_shared<pynative::AllocViewMemDeviceTask>(alloc_mem_func, device_context, tensor);
  auto &op_executor = runtime::OpExecutor::GetInstance();
  op_executor.PushSimpleOpRunTask(view_task);
}

void MindRTBackend::CompileSingleOpGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                         bool is_dynamic_shape) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  pynative::OpCompiler::GetInstance().BatchBuild({graph}, device_context, is_dynamic_shape);
}

void MindRTBackend::UpdateOutput(const session::BackendOpRunInfoPtr &op_run_info,
                                 const std::vector<session::KernelWithIndex> &output_nodes,
                                 VectorRef *const outputs) const {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(op_run_info);

  if (!op_run_info->device_sync_promises.empty()) {
    UpdateTensorAddress(op_run_info, output_nodes, outputs);
    return;
  }

  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto output_tensor = CreateOutputTensor(item_with_index.first, item_with_index.second);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    outputs->emplace_back(output_tensor);
  }
}

void MindRTBackend::UpdateOutputDynamic(const session::BackendOpRunInfoPtr &op_run_info,
                                        const OpCompilerInfoPtr &op_compiler_info,
                                        const vector<device::DeviceAddressPtr> &device_address_list,
                                        VectorRef *const outputs) const {
  MS_EXCEPTION_IF_NULL(op_run_info);

  if (!op_run_info->device_sync_promises.empty()) {
    MS_LOG(DEBUG) << "Has promise and update tensor address, op " << op_run_info->base_op_run_info.op_name;
    UpdateTensorAddressDynamic(op_run_info, op_compiler_info, device_address_list, outputs);
    return;
  }
  MS_LOG(DEBUG) << "No promise, just create tensor and address, op " << op_run_info->base_op_run_info.op_name;
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto output_nodes = op_compiler_info->graph_output_nodes_;
  auto outputs_size = output_nodes.size();
  if (op_compiler_info->graph_outputs_tensor_num_.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of graph_outputs_tensor_num_:" << op_compiler_info->graph_outputs_tensor_num_.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  if (device_address_list.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of device_address_list:" << device_address_list.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  for (size_t i = 0; i < outputs_size; ++i) {
    auto item_with_index = output_nodes[i];
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (op_compiler_info->graph_outputs_tensor_num_[i] == 0) {
      continue;
    }
    auto output_address = device_address_list[i];
    MS_EXCEPTION_IF_NULL(output_address);
    TensorPtr output_tensor =
      CreateOutputTensorDynamicImpl(op_compiler_info, item_with_index.first, item_with_index.second, output_address, i);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
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
