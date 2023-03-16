
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
#include "backend/common/session/session_basic.h"

#include <algorithm>
#include <set>
#include <queue>
#include <utility>
#include <functional>
#include <unordered_map>

#include "utils/hash_map.h"
#include "ops/primitive_c.h"
#include "ir/manager.h"
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "base/base_ref_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/config_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/executor_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/ms_utils.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "utils/file_utils.h"
#include "utils/trace_base.h"
#include "include/common/utils/parallel_context.h"
#include "kernel/oplib/oplib.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "abstract/abstract_value.h"
#endif
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#endif
#include "include/backend/debug/debugger/proto_exporter.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_exec_order_recorder.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/graph_recorder.h"
#include "runtime/hardware/device_context_manager.h"
#endif
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#endif

namespace mindspore {
namespace session {
MS_REG_SESSION(kSessionBasic, SessionBasic);

namespace {
constexpr int kSummaryGetItem = 2;
constexpr int64_t kInvalidShape = -2;
static bool IsPynativeMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
}

BaseRef GetNodeOutputTensorFromInputs(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                                      const std::vector<tensor::TensorPtr> &input_tensors) {
  auto &node = node_output_pair.first;
  MS_EXCEPTION_IF_NULL(node);
  if (HasAbstractMonad(node)) {
    return std::make_shared<tensor::Tensor>(int64_t(0), kBool);
  }
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  if (IsPynativeMode()) {
    return nullptr;
  }
  if (!node->isa<Parameter>()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_node = node->cast<ParameterPtr>();
  if (param_node != nullptr && param_node->IsUsedByRealKernelInGraph(graph->graph_id())) {
    return nullptr;
  }
  for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
    if (input_idx >= input_tensors.size()) {
      MS_LOG(EXCEPTION) << "Input idx:" << input_idx << " is out of range:" << input_tensors.size();
    }
    if (graph->inputs()[input_idx] == node) {
      return input_tensors[input_idx];
    }
  }
  return nullptr;
}

BaseRef CreateNodeOutputTensor(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                               const std::vector<tensor::TensorPtr> &input_tensors,
                               std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto &node = node_output_pair.first;
  size_t output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto tensor_from_input = GetNodeOutputTensorFromInputs(node_output_pair, graph, input_tensors);
  if (tensor_from_input != nullptr) {
    return tensor_from_input;
  }
  TypeId type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (type_id == kTypeUnknown) {
    type_id = common::AnfAlgo::GetOutputInferDataType(node, output_index);
  }

  auto shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    auto max_shape = common::AnfAlgo::GetOutputMaxShape(node, output_index);
    if (abstract::ShapeSize(max_shape) > abstract::ShapeSize(shape)) {
      shape = max_shape;
    }
  }
  tensor::TensorPtr tensor;
  bool is_internal_output = graph->IsInternalOutput(node, output_index);
  if (is_internal_output) {
    tensor = graph->GetInternalOutputTensor(node, output_index);
    if (tensor == nullptr) {
      tensor = std::make_shared<tensor::Tensor>(type_id, shape);
      graph->AddInternalOutputTensor(node, output_index, tensor);
    }
  } else {
    tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  }
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
  if (is_internal_output) {
    tensor->set_sync_status(kNoNeedSync);
  } else {
    // if in pynative mode,data only copied to host when user want to print data
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
        ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
      tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
    } else {
      tensor->set_sync_status(kNeedSyncDeviceToHost);
    }
  }
  tensor->SetIsGraphOutput();
  (*tensor_to_node)[tensor] = node_output_pair;
  return tensor;
}

std::string GetOpRunDeviceTarget(const PrimitivePtr &op_prim) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const std::string &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find(kAttrPrimitiveTarget);
  if (iter != attr_map.end()) {
    return GetValue<std::string>(iter->second);
  }
  return device_target;
}

// Need to discard input tensor properties in heterogeneous scenarios.
// For example, the format of device_address in input_tensor is 5D format,
// and it's invalid for CPU graph parameter.
bool NeedDiscardTensorProperties(const std::string &op_device_target,
                                 const device::DeviceAddressPtr &tensor_device_address) {
  if (tensor_device_address == nullptr) {
    return true;
  }

  if (op_device_target == device::GetDeviceNameByType(tensor_device_address->GetDeviceType())) {
    return false;
  }
  return true;
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     const BackendOpRunInfoPtr &op_run_info, int64_t tensor_mask) {
  MS_EXCEPTION_IF_NULL(graph);
  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  if (tensor_mask == kParameterWeightTensorMask) {
    param->set_default_param(input_tensor);
  }

  // set the kernel info of parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  if (NeedDiscardTensorProperties(op_run_info->base_op_run_info.device_target, device_address)) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = common::AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{param_init_data_type});
  } else {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{device_address->format()});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{device_address->type_id()});
    kernel_build_info_builder->SetOutputsReshapeType({input_tensor->padding_type()});
    AnfAlgo::SetOutputAddr(device_address, 0, param.get());
  }
  if (input_tensor->isa<tensor::MapTensor>()) {
    auto map_tensor = input_tensor->cast<tensor::MapTensorPtr>();
    auto map_tensor_abs = std::make_shared<abstract::AbstractMapTensor>(map_tensor);
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
    param->set_abstract(map_tensor_abs);
    return param;
  }
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
  // construct abstract of parameter
  auto type_of_tensor = input_tensor->Dtype();
  std::shared_ptr<abstract::AbstractTensor> abstract;
  // Base_shape_ptr is set in dynamic shape scenario, if nullptr, not dynamic shape
  if (input_tensor->base_shape_ptr() != nullptr) {
    abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, input_tensor->base_shape_ptr());
  } else {
    abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, input_tensor->shape());
  }
  param->set_abstract(abstract);
  return param;
}

void DumpGraphOutput(const Any &any, size_t recurse_level = 0) {
  MS_LOG(INFO) << "Graph outputs:";
  const size_t max_deep = 10;
  if (recurse_level > max_deep) {
    MS_LOG(INFO) << "Recurse too deep";
    return;
  }
  std::string tab_str;
  for (size_t i = 0; i < recurse_level; i++) {
    tab_str = tab_str.append("  ");
  }
  if (any.is<AnyList>()) {
    (void)tab_str.append("{");
    MS_LOG(INFO) << tab_str;
    auto any_list = any.cast<AnyList>();
    for (auto &it : any_list) {
      DumpGraphOutput(it, recurse_level + 1);
    }
    (void)tab_str.append("}");
    MS_LOG(INFO) << tab_str;
  }
  (void)tab_str.append(any.ToString());
  MS_LOG(INFO) << tab_str;
}

BaseRef CreateNodeOutputPlaceholder(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                                    const std::vector<tensor::TensorPtr> &input_tensors,
                                    const std::vector<size_t> &indexes,
                                    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  auto &node = node_output_pair.first;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(output_indexes);
  MS_LOG(DEBUG) << "Create placeholder for output[" << node->DebugString() << "] index[" << node_output_pair.second
                << "]";
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  if (node->isa<Parameter>()) {
    const auto &input_nodes = graph->input_nodes();
    for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
      if (input_idx >= input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Input idx:" << input_idx << " is out of range:" << input_tensors.size();
      }
      if (input_nodes[input_idx] == node) {
        return input_tensors[input_idx];
      }
    }
    MS_LOG(EXCEPTION) << "Parameter: " << node->DebugString() << " has no output addr";
  }
  (*output_indexes)[node_output_pair].emplace_back(indexes);
  BaseRef output_placeholder = std::make_shared<BaseRef>();
  return output_placeholder;
}

BaseRef CreateNodeOutputPlaceholder(const AnfNodePtr &anf, const KernelGraphPtr &graph,
                                    const std::vector<tensor::TensorPtr> &input_tensors,
                                    const std::vector<size_t> &indexes,
                                    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(output_indexes);
  MS_LOG(DEBUG) << "Create placeholder for output[" << anf->DebugString() << "]";
  auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(DEBUG) << "Create placeholder for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (common::AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      std::vector<size_t> cur_index = indexes;
      cur_index.emplace_back(i - 1);
      auto out = CreateNodeOutputPlaceholder(cnode->input(i), graph, input_tensors, cur_index, output_indexes);
      ret.push_back(out);
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }
  return CreateNodeOutputPlaceholder(item_with_index, graph, input_tensors, indexes, output_indexes);
}

void CheckInputTensorShape(const TensorPtr &tensor, const CNodePtr &kernel, size_t input_index) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &tensor_shape = tensor->shape();
  const auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel, input_index);
  if (tensor_shape.size() != input_shape.size()) {
    MS_LOG(EXCEPTION) << "The input tensor's shape size: " << tensor_shape.size()
                      << " is not equal to expected size: " << input_shape.size() << " for input[" << input_index
                      << "] of kernel: " << common::AnfAlgo::GetCNodeName(kernel) << trace::DumpSourceLines(kernel);
  }
  for (size_t i = 0; i < tensor_shape.size(); i++) {
    if (tensor_shape[i] < 0 || (tensor_shape[i] != input_shape[i] && input_shape[i] >= 0)) {
      MS_LOG(EXCEPTION) << "The input tensor's shape: " << tensor_shape
                        << " is not equal to expected shape: " << input_shape << " for input[" << input_index
                        << "] of kernel: " << common::AnfAlgo::GetCNodeName(kernel) << trace::DumpSourceLines(kernel);
    }
  }
}

void IterateFindTensor(std::vector<ValuePtr> *msTensors, const VectorRef &ref_list) {
  MS_EXCEPTION_IF_NULL(msTensors);
  for (size_t i = 0; i < ref_list.size(); ++i) {
    if (utils::isa<tensor::TensorPtr>(ref_list[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(ref_list[i]);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      msTensors->emplace_back(tensor_ptr);
    } else if (utils::isa<VectorRef>(ref_list[i])) {
      auto ref_iter = utils::cast<VectorRef>(ref_list[i]);
      IterateFindTensor(msTensors, ref_iter);
    } else if (utils::isa<tensor::CSRTensorPtr>(ref_list[i])) {
      auto csr_tensor = utils::cast<tensor::CSRTensorPtr>(ref_list[i]);
      MS_EXCEPTION_IF_NULL(csr_tensor);
      (void)msTensors->emplace_back(csr_tensor);
    } else {
      MS_LOG(EXCEPTION) << "The output is not a tensor/sparse tensor";
    }
  }
}

std::vector<ValuePtr> TransformVectorRefToMultiValue(const VectorRef &base_ref) {
  std::vector<ValuePtr> msTensors;
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    IterateFindTensor(&msTensors, ref_list);
  } else if (utils::isa<tensor::Tensor>(base_ref)) {
    auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    (void)msTensors.emplace_back(tensor_ptr);
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
  }
  return msTensors;
}

bool is_param_scalar(const size_t &param_shape_size, const size_t &input_shape_size) {
  if (param_shape_size == 1 && input_shape_size == 0) {
    return true;
  }
  if (param_shape_size == 0 && input_shape_size == 1) {
    return true;
  }
  return false;
}
}  // namespace

BaseRef SessionBasic::CreateNodeOutputTensors(const AnfNodePtr &anf, const KernelGraphPtr &graph,
                                              const std::vector<tensor::TensorPtr> &input_tensors,
                                              std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                                              KernelMapTensor *node_to_tensor) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_EXCEPTION_IF_NULL(node_to_tensor);
  MS_LOG(DEBUG) << "Create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(DEBUG) << "Create tensor for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (common::AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto out = CreateNodeOutputTensors(cnode->input(i), graph, input_tensors, tensor_to_node, node_to_tensor);
      ret.push_back(out);
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }

  //  The outputs of graph may have the same kernel node, no need to create new tensor.
  const auto &iter = node_to_tensor->find(item_with_index);
  if (iter != node_to_tensor->end()) {
    return iter->second;
  }

  const auto &tensor = CreateNodeOutputTensor(item_with_index, graph, input_tensors, tensor_to_node);
  (*node_to_tensor)[item_with_index] = tensor;
  return tensor;
}

void SessionBasic::InitExecutor(const std::string &device_name, uint32_t device_id) {
  device_id_ = device_id;
  context_ = std::make_shared<Context>(device_name, device_id);
  executor_ = ExecutorManager::Instance().GetExecutor(device_name, device_id);
}

void SessionBasic::GetSingleOpGraphInfo(const CNodePtr &kernel, const InputTensorInfo &tensor_info,
                                        GraphInfo *graph_info, const BackendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get input tensor info
  const auto &input_tensors = tensor_info.input_tensors;
  const auto &input_tensors_mask = tensor_info.input_tensors_mask;
  if (input_tensors.size() != input_tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << input_tensors_mask.size();
  }

  std::ostringstream buf;
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  buf << GetOpRunDeviceTarget(prim) << "_dynamic" << op_run_info->base_op_run_info.use_dynamic_shape_process << "_";
  buf << prim->name() << "_";
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto &tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    bool use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
    if (use_dynamic_shape_process) {
      buf << tensor->shape().size() << "_";
    } else {
      if (tensor->base_shape_ptr() != nullptr) {
        buf << tensor->base_shape_ptr()->ToString();
      } else {
        buf << tensor->shape();
      }
    }

    buf << tensor->data_type();
    buf << tensor->padding_type();
    // In the case of the same shape, but dtype and format are inconsistent
    if (tensor->device_address() != nullptr) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
      MS_EXCEPTION_IF_NULL(p_address);
      buf << p_address->type_id();
      buf << p_address->format();
    }
    // For constant input
    if (input_tensors_mask[i] == kValueNodeTensorMask) {
      buf << common::AnfAlgo::GetTensorValueString(tensor);
    }
    buf << "_";
  }

  // Get attr info
  const auto &attr_map = prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&buf](const auto &element) { buf << element.second->ToString(); });

  *graph_info = buf.str();
}

BackendOpRunInfoPtr SessionBasic::GetSingleOpRunInfo(const CNodePtr &cnode, const GraphInfo &graph_info,
                                                     const InputTensorInfo &tensor_info,
                                                     const GraphOutputInfo *const graph_output_info) const {
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  const auto &abstract = cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract is nullptr, node = " << cnode->DebugString();
  }
  const auto &shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);

  bool is_gradient_out = false;
  if (graph_output_info != nullptr) {
    auto lb_iter = graph_output_info->output_indexes.lower_bound({cnode, 0});
    is_gradient_out = lb_iter != graph_output_info->output_indexes.end() && lb_iter->first.first == cnode;
  }
  pynative::BaseOpRunInfo base_op_run_info;
  base_op_run_info.is_mixed_precision_cast = false;
  base_op_run_info.lazy_build = !shape->IsDynamic();
  base_op_run_info.op_name = primitive->name();
  base_op_run_info.next_op_name = std::string();
  base_op_run_info.graph_info = graph_info;
  base_op_run_info.device_target = GetOpRunDeviceTarget(primitive);
  base_op_run_info.next_input_index = 0;
  base_op_run_info.input_tensor = tensor_info.input_tensors;
  base_op_run_info.input_mask = tensor_info.input_tensors_mask;
  base_op_run_info.abstract = abstract;
  return std::make_shared<BackendOpRunInfo>(base_op_run_info, primitive, false, is_gradient_out);
}

void SessionBasic::GetParameterIndex(const KernelGraph *graph, const std::vector<tensor::TensorPtr> &inputs,
                                     std::map<AnfNodePtr, size_t> *parameter_index) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter_index);
  size_t index = 0;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  bool is_parallel_forward_ms_function =
    !graph->has_flag(kFlagIsPynativeBpropGraph) &&
    (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel);
  for (const auto &input_node : graph->input_nodes()) {
    auto params = common::AnfAlgo::GetAllOutput(input_node);
    for (const auto &param : params) {
      if (index >= inputs.size()) {
        MS_LOG(EXCEPTION) << "Parameter size out of range. Parameter index: " << index
                          << ", input size: " << inputs.size();
      }
      const auto &input = inputs[index];
      MS_EXCEPTION_IF_NULL(input);
      MS_EXCEPTION_IF_NULL(param);
      // Check shape of input and parameter
      const auto &input_shape = input->shape();
      const auto &param_shape = common::AnfAlgo::GetOutputInferShape(param, 0);
      bool is_dynamic = param->Shape()->IsDynamic();
      // Dynamic shape feed mode, shape is dynamic but max shape is ()
      if (!is_dynamic || !param_shape.empty()) {
        if (!is_parallel_forward_ms_function && input_shape.size() != param_shape.size()) {
          // Infer shape is -2, which indicates that the shape cannot be infer currently
          if (param_shape.size() == 1 && param_shape[0] == kInvalidShape) {
            parameter_index->emplace(param, index++);
            continue;
          }
          // Input is scalar. param shape will be [1], input shape will be []
          if (is_param_scalar(param_shape.size(), input_shape.size())) {
            parameter_index->emplace(param, index++);
            continue;
          }
          MS_LOG(EXCEPTION) << "Shape size of input tensor(" << input_shape << ") and parameter(" << param_shape
                            << ") are different, input index: " << index << ", parameter: " << param->DebugString();
        }
        for (size_t i = 0; i < input_shape.size(); i += 1) {
          if (input_shape[i] < 0 ||
              (!is_parallel_forward_ms_function && input_shape[i] != param_shape[i] && !is_dynamic)) {
            MS_LOG(EXCEPTION) << "Input tensor shape(" << input_shape << ") and parameter shape(" << param_shape
                              << ") are different, input index: " << index << ", parameter: " << param->DebugString();
          }
        }
      }
      parameter_index->emplace(param, index++);
    }
  }
}

void SessionBasic::CreateOutputPlaceholder(
  const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *const outputs,
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_indexes);
  auto anf_outputs = kernel_graph->outputs();
  size_t index = 0;
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    std::vector<size_t> indexes{index++};
    outputs->emplace_back(CreateNodeOutputPlaceholder(item, kernel_graph, input_tensors, indexes, output_indexes));
  }
}

void SessionBasic::GetRefCount(const KernelGraph *graph, std::map<KernelWithIndex, size_t> *ref_count) const {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &kernel : graph->execution_order()) {
    for (size_t i = 1; i < kernel->inputs().size(); i += 1) {
      const auto &input = kernel->input(i);
      auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
      const auto &node = kernel_with_index.first;
      if (node->isa<CNode>()) {
        (*ref_count)[kernel_with_index] += 1;
      }
    }
  }
}

void SessionBasic::GetForwardOpOutputRefCount(const KernelGraph *graph, const std::vector<tensor::TensorPtr> &inputs,
                                              std::map<std::string, size_t> *forward_op_output_tensor_id,
                                              const std::map<AnfNodePtr, size_t> &parameter_index) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // Cpu can not clear device address, because it's device address and host address is the same
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    return;
  }
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    const auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 1; i <= input_tensor_num; ++i) {
      const auto &input = kernel->input(i);
      auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
      auto real_input = kernel_with_index.first;
      MS_EXCEPTION_IF_NULL(real_input);
      if (real_input->isa<ValueNode>()) {
        const auto &tensor = GetValueNodeOutputTensor(real_input, kernel_with_index.second);
        if (tensor == nullptr) {
          continue;
        }
        if (tensor->is_forward_output()) {
          (*forward_op_output_tensor_id)[tensor->id()] += 1;
        }
      } else if (real_input->isa<Parameter>()) {
        // Forward op output use as sens, so need add reference
        auto iter = parameter_index.find(real_input);
        if (iter != parameter_index.end()) {
          auto tensor = inputs[iter->second];
          if (tensor->is_forward_output()) {
            (*forward_op_output_tensor_id)[tensor->id()] += 1;
          }
        }
      }
    }
  }
  MS_LOG(DEBUG) << "Forward op output tensor in bprop graph size " << forward_op_output_tensor_id->size();
}

void SessionBasic::ReleaseForwardOpOutput(const std::vector<tensor::TensorPtr> &input_tensors,
                                          std::map<std::string, size_t> *forward_op_output_tensor_id) const {
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  for (const auto &tensor : input_tensors) {
    auto it = forward_op_output_tensor_id->find(tensor->id());
    if (it != forward_op_output_tensor_id->end()) {
      if (--(it->second) == 0) {
        tensor->set_device_address(nullptr);
        forward_op_output_tensor_id->erase(it);
      }
    }
  }
}

void SessionBasic::HandleOpInputs(const std::set<KernelWithIndex> &input_kernel,
                                  std::map<KernelWithIndex, size_t> *ref_count,
                                  std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) const {
  MS_EXCEPTION_IF_NULL(ref_count);
  MS_EXCEPTION_IF_NULL(op_output_map);
  for (const auto &kernel_with_index : input_kernel) {
    if (!kernel_with_index.first->isa<CNode>()) {
      continue;
    }

    // Release previous output
    auto ref_iter = ref_count->find(kernel_with_index);
    if (ref_iter == ref_count->end()) {
      MS_LOG(EXCEPTION) << "Can not find input KernelWithIndex in cnode reference count map, input cnode = "
                        << kernel_with_index.first->DebugString() << ", index = " << kernel_with_index.second;
    }
    // Reduce reference count number, when it was reduced to zero, release the useless output of pre node.
    ref_iter->second -= 1;
    if (ref_iter->second != 0) {
      continue;
    }
    ref_count->erase(ref_iter);
    auto output_iter = op_output_map->find(kernel_with_index);
    if (output_iter == op_output_map->end()) {
      MS_LOG(EXCEPTION) << "Can not find input KernelWithIndex in op_output map, input cnode = "
                        << kernel_with_index.first->DebugString() << ", index = " << kernel_with_index.second;
    }
    op_output_map->erase(output_iter);
  }
}

void SessionBasic::HandleOpOutputs(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                                   const std::map<KernelWithIndex, size_t> &ref_count,
                                   std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map,
                                   GraphOutputInfo *const graph_output_info) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_output_map);
  MS_EXCEPTION_IF_NULL(graph_output_info);
  MS_EXCEPTION_IF_NULL(graph_output_info->graph_outputs);
  auto output_values = TransformVectorRefToMultiValue(op_outputs);
  if (output_values.size() > op_outputs.size()) {
    MS_LOG(EXCEPTION) << "Op output contains tuple, node = " << kernel->DebugString();
  }
  size_t out_index = 0;
  for (const auto &output_value : output_values) {
    auto kernel_with_index = make_pair(kernel, out_index++);
    auto output_tensor = output_value->cast<tensor::TensorPtr>();
    bool value_is_tensor = (output_tensor != nullptr);
    if (ref_count.find(kernel_with_index) != ref_count.end() && value_is_tensor) {
      (*op_output_map)[kernel_with_index] = output_tensor;
    }
    const auto &iter = graph_output_info->output_indexes.find(kernel_with_index);
    if (iter == graph_output_info->output_indexes.end()) {
      continue;
    }
    const std::vector<std::vector<size_t>> &multiple_ref_indexes = iter->second;
    for (const auto &ref_indexes : multiple_ref_indexes) {
      size_t n = 0;
      const VectorRef *cur_vector_ref = graph_output_info->graph_outputs;
      for (; n < ref_indexes.size() - 1; n += 1) {
        size_t index = ref_indexes.at(n);
        if (index >= cur_vector_ref->size()) {
          MS_LOG(EXCEPTION) << "Get invalid output ref index: " << index << ", size of vertor ref is "
                            << cur_vector_ref->size();
        }
        const BaseRef &base_ref = (*cur_vector_ref)[index];
        if (!utils::isa<VectorRef>(base_ref)) {
          MS_LOG(EXCEPTION) << "Get none VectorRef by ref index, index: " << index << "cur n: " << n;
        }
        cur_vector_ref = &utils::cast<VectorRef>(base_ref);
      }
      BaseRef &tensor_ref = (*const_cast<VectorRef *>(cur_vector_ref))[ref_indexes.at(n)];
      tensor_ref = output_value;
      if (value_is_tensor) {
        graph_output_info->graph_output_tensors.emplace_back(output_tensor);
      }
    }
  }
}

TensorPtr SessionBasic::GetValueNodeOutputTensor(const AnfNodePtr &node, size_t output_index) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = GetValueNode(value_node);
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    if (output_index >= value_tuple->size()) {
      MS_LOG(EXCEPTION) << "Index " << output_index << "is out of value tuple range";
    }
    auto tensor_value = value_tuple->value()[output_index];
    if (tensor_value->isa<tensor::Tensor>()) {
      return tensor_value->cast<tensor::TensorPtr>();
    }
  } else if (value->isa<tensor::Tensor>()) {
    if (output_index != 0) {
      MS_LOG(EXCEPTION) << "Index should be 0 for Tensor ValueNode, but is " << output_index;
    }
    return value->cast<TensorPtr>();
  } else if (value->isa<StringImm>()) {
    auto value_string = GetValue<std::string>(value);
    const ShapeVector shape = {1, SizeToLong(value_string.size())};
    TensorPtr tensor = std::make_shared<Tensor>(kObjectTypeString, shape, value_string.data(), value_string.size());
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_sync_status(kNeedSyncHostToDevice);
    return tensor;
  } else if (value->isa<tensor::CSRTensor>()) {
    return value->cast<tensor::CSRTensorPtr>()->GetTensorAt(output_index);
  } else if (value->isa<tensor::COOTensor>()) {
    return value->cast<tensor::COOTensorPtr>()->GetTensorAt(output_index);
  } else if (value->isa<Type>()) {
    int64_t type_id = value->cast<TypePtr>()->type_id();
    auto tensor = std::make_shared<tensor::Tensor>(type_id, kInt64);
    tensor->set_user_data(kTensorValueIsType, value);
    return tensor;
  }
  return nullptr;
}

TensorPtr SessionBasic::GetParameterOutputTensor(const AnfNodePtr &node,
                                                 const std::map<AnfNodePtr, size_t> &parameter_index,
                                                 const std::vector<tensor::TensorPtr> &graph_inputs) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<Parameter>()) {
    return nullptr;
  }
  const auto &iter = parameter_index.find(node);
  if (iter == parameter_index.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter input of cnode, parameter = " << node->DebugString();
  }
  const size_t index = iter->second;
  if (index >= graph_inputs.size()) {
    MS_LOG(EXCEPTION) << "Parameter index is greater than size of graph's input tensor, parameter index = " << index
                      << ", input tensor size = " << graph_inputs.size();
  }
  return graph_inputs[index];
}

TensorPtr SessionBasic::GetCNodeOutputTensor(const KernelWithIndex &kernel_with_index,
                                             const std::map<KernelWithIndex, tensor::TensorPtr> &op_output) const {
  const auto &iter = op_output.find(kernel_with_index);
  if (iter == op_output.end()) {
    MS_LOG(EXCEPTION) << "Can not find output tensor of cnode, node = " << kernel_with_index.first->DebugString();
  }
  return iter->second;
}

void SessionBasic::GetConstValueDepend(const CNodePtr &cnode, std::vector<size_t> *const_input_attr_index) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(const_input_attr_index);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_adaptation_info =
    opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, kAscendDevice, true);
  if (op_adaptation_info == nullptr) {
    MS_LOG(DEBUG) << "Cannot get op_adaptation_info for " << op_name << " no need to convert input to attr.";
    return;
  }
  // No need to convert input to attr for dynamic op.
  if (op_adaptation_info->need_tbe_check_supported() && common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
    return;
  }
  auto input_to_attr_map = op_adaptation_info->input_attr_map();
  std::transform(input_to_attr_map.begin(), input_to_attr_map.end(), std::back_inserter(*const_input_attr_index),
                 [](auto iter) { return iter.first; });
}

void SessionBasic::GetOpInputTensors(const CNodePtr &cnode,
                                     const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                                     const std::map<AnfNodePtr, size_t> &parameter_index,
                                     const std::vector<tensor::TensorPtr> &graph_inputs,
                                     InputTensorInfo *input_tensor_info) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_tensor_info);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<size_t> const_input_attr_index = {};
  GetConstValueDepend(cnode, &const_input_attr_index);
  MS_LOG(DEBUG) << "const_input_attr_index " << const_input_attr_index;
  const auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 1; i <= input_tensor_num; i += 1) {
    const auto &input = cnode->input(i);
    auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
    auto real_input = kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    tensor::TensorPtr tensor = nullptr;
    if (real_input->isa<ValueNode>()) {
      tensor = GetValueNodeOutputTensor(real_input, kernel_with_index.second);
      const auto &value_ptr = GetValueNode(real_input);
      MS_EXCEPTION_IF_NULL(value_ptr);
      auto is_value_node = value_ptr->isa<StringImm>();
      if (!const_input_attr_index.empty()) {
        is_value_node = std::find(const_input_attr_index.begin(), const_input_attr_index.end(), i - 1) !=
                        const_input_attr_index.end();
      }

      bool is_forward_output = false;
      if (value_ptr->isa<tensor::Tensor>()) {
        auto forward_tensor = value_ptr->cast<tensor::TensorPtr>();
        if (forward_tensor->is_forward_output()) {
          is_forward_output = true;
        }
      }

      if (common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
        input_tensor_info->input_tensors_mask.emplace_back(
          (is_value_node && !is_forward_output) ? kValueNodeTensorMask : kParameterDataTensorMask);
      } else {
        input_tensor_info->input_tensors_mask.emplace_back(
          (is_value_node || !is_forward_output) ? kValueNodeTensorMask : kParameterDataTensorMask);
      }
    } else if (real_input->isa<Parameter>()) {
      tensor = GetParameterOutputTensor(real_input, parameter_index, graph_inputs);
      input_tensor_info->input_tensors_mask.emplace_back(tensor->is_parameter() ? kParameterWeightTensorMask
                                                                                : kParameterDataTensorMask);
    } else if (real_input->isa<CNode>()) {
      tensor = GetCNodeOutputTensor(kernel_with_index, op_output);
      if (common::AnfAlgo::IsControlOpExecInBackend(real_input)) {
        CheckInputTensorShape(tensor, cnode, i - 1);
      }
      input_tensor_info->input_kernel.insert(kernel_with_index);
      input_tensor_info->input_tensors_mask.emplace_back(tensor->is_parameter() ? kParameterWeightTensorMask
                                                                                : kParameterDataTensorMask);
    } else {
      MS_LOG(EXCEPTION) << "Invalid input node, node = " << real_input->DebugString();
    }
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Get" << i << "th input tensor of " << cnode->fullname_with_scope() << " from "
                  << real_input->fullname_with_scope() << "-" << kernel_with_index.second;
    BaseShapePtr base_shape = nullptr;
    auto real_input_abs = real_input->abstract();
    MS_EXCEPTION_IF_NULL(real_input_abs);
    if (real_input_abs->isa<abstract::AbstractTuple>()) {
      auto tuple_abs = real_input_abs->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abs);
      auto tuple_abs_elem = tuple_abs->elements()[kernel_with_index.second];
      MS_EXCEPTION_IF_NULL(tuple_abs_elem);
      base_shape = tuple_abs_elem->BuildShape();
    } else {
      base_shape = real_input_abs->BuildShape();
    }
    MS_EXCEPTION_IF_NULL(base_shape);
    if (base_shape->IsDynamic()) {
      tensor->set_base_shape(base_shape);
    }
    input_tensor_info->input_tensors.emplace_back(tensor);
  }
}

tensor::TensorPtr SessionBasic::GetOpInputTensorByIndex(const CNodePtr &cnode,
                                                        const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                                                        const std::map<AnfNodePtr, size_t> &parameter_index,
                                                        const std::vector<tensor::TensorPtr> &graph_inputs,
                                                        InputTensorInfo *input_tensor_info, size_t input_index) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_tensor_info);
  if (input_index >= cnode->inputs().size() - 1) {
    MS_LOG(EXCEPTION) << "Input index is out of range:" << cnode->inputs().size() << ",cnode:" << cnode->DebugString();
  }

  const auto &input = cnode->input(input_index + 1);
  auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
  auto real_input = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input);

  if (real_input->isa<Parameter>()) {
    return GetParameterOutputTensor(real_input, parameter_index, graph_inputs);
  } else if (real_input->isa<CNode>()) {
    tensor::TensorPtr tensor = GetCNodeOutputTensor(kernel_with_index, op_output);
    if (common::AnfAlgo::IsControlOpExecInBackend(real_input)) {
      CheckInputTensorShape(tensor, cnode, input_index);
    }
    input_tensor_info->input_kernel.insert(kernel_with_index);
    return tensor;
  } else {
    MS_LOG(EXCEPTION) << "Invalid input node, node = " << real_input->DebugString();
  }
}

void SessionBasic::UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                                 const std::vector<tensor::TensorPtr> &input_tensors,
                                 std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  KernelMapTensor node_to_tensor;
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(DEBUG) << "Update output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, tensor_to_node, &node_to_tensor));
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (auto &item : *tensor_to_node) {
    auto &tensor = item.first;
    auto &node = item.second.first;
    auto &output_index = item.second.second;
    DeviceAddressPtr address = nullptr;
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode &&
        ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
      address = AnfAlgo::GetMutableOutputAddr(node, output_index, false);
    } else {
      address = AnfAlgo::GetMutableOutputAddr(node, output_index);
    }
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(address);
    tensor->SetNeedWait(false);
    MS_LOG(DEBUG) << "Debug address: Output tensor obj " << tensor.get() << ", tensor id " << tensor->id()
                  << ", device address " << tensor->device_address().get();
    if (common::AnfAlgo::IsDynamicShape(node)) {
      const auto &updated_shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
      (void)tensor->set_shape(updated_shape);
    }
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
      tensor->data_sync(false);
      tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

std::vector<tensor::TensorPtr> SessionBasic::GetInputNeedLockTensors(
  const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs) const {
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_optimizer()) {
    return {};
  }
  auto input_nodes = graph->inputs();
  bool check_monad = false;
  if (input_nodes.size() == inputs.size()) {
    check_monad = true;
  }
  std::vector<tensor::TensorPtr> result;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (check_monad && HasAbstractMonad(input_nodes[i])) {
      continue;
    }
    auto &tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    if (!tensor->IsGraphOutput()) {
      result.emplace_back(tensor);
    }
  }
  return result;
}

void SessionBasic::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                       VectorRef *outputs,
                                       std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                                       KernelMapTensor *node_to_tensor) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, tensor_to_node, node_to_tensor));
  }
}

void SessionBasic::UpdateOutputTensors(const VectorRef *outputs,
                                       const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                       std::map<DeviceAddressPtr, DeviceAddressPtr> *) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (device::KernelRuntime::UseMemScheduler()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(outputs);
  for (const auto &item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      const auto &vector_ref = utils::cast<VectorRef>(item);
      std::map<DeviceAddressPtr, DeviceAddressPtr> new_to_old_device_address;
      UpdateOutputTensors(&vector_ref, tensor_to_node, &new_to_old_device_address);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      const auto &tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      const auto &iter = tensor_to_node.find(tensor);
      if (iter != tensor_to_node.end()) {
        const auto &node = iter->second.first;
        const auto &output_index = iter->second.second;
        if (!AnfAlgo::OutputAddrExist(node, output_index, true)) {
          continue;
        }
        const auto &address = AnfAlgo::GetMutableOutputAddr(node, output_index);
        tensor->set_device_address(address);

        if (common::AnfAlgo::IsDynamicShape(node)) {
          const auto &updated_shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
          (void)tensor->set_shape(updated_shape);
        }
      }
      if (tensor->NeedSyncDeviceToHostImmediately()) {
        tensor->data_sync(false);
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
      }
    }
  }
}

void SessionBasic::GetModelInputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *inputs,
                                      std::vector<std::string> *inputs_name) const {
  MS_LOG(INFO) << "Start get model inputs, graph id : " << graph_id;
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(inputs_name);
  auto kernel_graph_inputs = kernel_graph->inputs();
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    if (!kernel_graph_inputs[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      continue;
    }
    auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
    if (!common::AnfAlgo::IsParameterWeight(parameter)) {
      auto input_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
      auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
      auto data_type = kernel_build_info->GetOutputDeviceType(0);
      auto ms_tensor = std::make_shared<tensor::Tensor>(data_type, input_shape);
      inputs->push_back(ms_tensor);
      inputs_name->push_back(parameter->name());
    }
  }
}

void SessionBasic::GetModelOutputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *outputs,
                                       std::vector<std::string> *output_names) const {
  std::vector<tensor::TensorPtr> inputs;
  std::vector<std::string> input_names;
  GetModelInputsInfo(graph_id, &inputs, &input_names);

  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_names);

  VectorRef vector_outputs;
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  KernelMapTensor node_to_tensor;
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    vector_outputs.emplace_back(CreateNodeOutputTensors(item, kernel_graph, inputs, &tensor_to_node, &node_to_tensor));
  }
  *outputs = TransformVectorRefToMultiTensor(vector_outputs);
  for (size_t i = 0; i < outputs->size(); i++) {
    output_names->push_back("output" + std::to_string(i));
  }
}

#ifndef ENABLE_SECURITY
void SessionBasic::RegisterSummaryCallBackFunc(const CallBackFunc &callback) {
  MS_EXCEPTION_IF_NULL(callback);
  summary_callback_ = callback;
}

void SessionBasic::SetSummaryNodesForAllGraphs(KernelGraph *graph, const std::vector<KernelGraphPtr> &all_graphs) {
  MS_LOG(DEBUG) << "Set summary nodes for all graphs start.";
  MS_EXCEPTION_IF_NULL(graph);
  auto summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.cbegin(), summary_nodes.cend());
  RecurseSetSummaryNodes(graph, all_graphs, &summary);
  graph->set_summary_nodes(summary);
  MS_LOG(INFO) << "The total summary nodes is: " << summary.size();
}

void SessionBasic::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }
  auto summary = graph->summary_nodes();
  auto apply_list = TopoSort(graph->get_return());
  for (auto &n : apply_list) {
    MS_EXCEPTION_IF_NULL(n);
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= kSummaryGetItem) {
        MS_LOG(EXCEPTION) << "The node Summary should have 2 inputs at least, but got " << (cnode->inputs().size() - 1)
                          << "." << trace::DumpSourceLines(cnode);
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (!AnfUtils::IsRealKernel(item_with_index.first)) {
        MS_LOG(EXCEPTION) << "Unexpected node:" << item_with_index.first->DebugString();
      }
      summary[n->fullname_with_scope()] = item_with_index;
    }
  }
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

void SessionBasic::RecurseSetSummaryNodes(KernelGraph *graph, std::vector<KernelGraphPtr> all_graphs,
                                          std::map<std::string, std::pair<AnfNodePtr, int>> *summary) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(summary);
  for (auto &child_graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(child_graph);
    SetSummaryNodes(child_graph.get());
    auto child_graph_summary = child_graph->summary_nodes();
    summary->insert(child_graph_summary.cbegin(), child_graph_summary.cend());
  }
  graph->set_summary_nodes(*summary);
}

void SessionBasic::Summary(KernelGraph *graph) {
  if (summary_callback_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  bool exist_summary = graph->summary_node_exist();
  if (!exist_summary) {
    return;
  }

  static bool is_first = true;
  if (is_first && !IsSupportSummary()) {
    is_first = false;
    MS_LOG(WARNING) << "The Summary operator can not collect data correctly. Detail: the data sink mode is used and the"
                       " sink size(in model.train() python api) is not equal to 1.";
  }
  SetSummaryNodes(graph);
  auto summary_outputs = graph->summary_nodes();
  std::map<std::string, tensor::TensorPtr> params_list;
  // fetch outputs apply kernel in session & run callback functions
  for (const auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetOutputAddr(node, index, false);
    auto shape = common::AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
    MS_EXCEPTION_IF_NULL(address);
    if (!address->GetPtr()) {
      continue;
    }
    if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, index), LongToSize(tensor->data().nbytes()),
                                   tensor->data_type(), tensor->data_c())) {
      MS_LOG(ERROR) << "Failed to sync output from device to host.";
    }
    tensor->set_sync_status(kNoNeedSync);
    params_list[output_item.first] = tensor;
  }
  // call callback function here
  summary_callback_(0, params_list);
}
#endif

void SessionBasic::CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) const {
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(std::make_shared<Primitive>(*prim::kPrimMakeTuple)));
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfAlgo::GetOutputElementNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfAlgo::GetOutputElementNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToLong(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int64Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      auto getitem = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(*prim::kPrimTupleGetItem)), cnode, idx});
      std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(cnode, output_index)};
      auto shapes = {common::AnfAlgo::GetOutputInferShape(cnode, output_index)};
      common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
      make_tuple_inputs.push_back(getitem);
    }
  } else {
    make_tuple_inputs.push_back(cnode);
  }
  // create output
  auto g_output = graph->NewCNode(make_tuple_inputs);
  graph->set_output(g_output);
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const BackendOpRunInfoPtr &op_run_info,
                                                                  const std::vector<tensor::TensorPtr> &input_tensors,
                                                                  const std::vector<int64_t> &tensors_mask,
                                                                  bool is_ascend) {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graph_sum_++;
  std::vector<AnfNodePtr> inputs;
  // set input[0]
  auto op_prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(op_prim);
  // Decoupling of frontend PrimitivePy and backend Primitive
  inputs.push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*op_prim)));
  // set input parameter
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (tensors_mask[i] == kValueNodeTensorMask) {
      auto value_node = graph->NewValueNode(input_tensors[i]);
      inputs.push_back(value_node);
      continue;
    }
    auto parameter = ConstructRunOpParameter(graph, input_tensors[i], op_run_info, tensors_mask[i]);
    inputs.push_back(parameter);
    auto mutable_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(mutable_inputs);
    mutable_inputs->push_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_mutable = common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode);
  if (is_mutable) {
    graph->set_flag(kAttrMutableKernel, true);
  }
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info->base_op_run_info.abstract);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(op_run_info->base_op_run_info.has_dynamic_output),
                               cnode);
  if (op_run_info->base_op_run_info.is_mixed_precision_cast) {
    common::AnfAlgo::SetNodeAttr(kAttrPynativeNextOpName, MakeValue(op_run_info->base_op_run_info.next_op_name), cnode);
    common::AnfAlgo::SetNodeAttr(kAttrPynativeNextIndex, MakeValue(op_run_info->base_op_run_info.next_input_index),
                                 cnode);
  }
  // set execution order
  std::vector<CNodePtr> exe_order = {cnode};
  graph->set_execution_order(exe_order);
  if (is_ascend && !is_mutable) {
    graph->set_output(cnode);
  } else {
    CreateOutputNode(cnode, graph);
  }
  graph->SetInputNodes();
  auto manager = MakeManager({graph});
  if (manager != nullptr) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    UnifyMindIR(graph);
  }
  graph->UpdateGraphDynamicAttr();
  return graph;
}

AnfNodePtr SessionBasic::FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list) const {
  MS_EXCEPTION_IF_NULL(push_node);
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      for (auto input : node->cast<CNodePtr>()->inputs()) {
        if (push_node == common::AnfAlgo::VisitKernel(input, 0).first) {
          if (common::AnfAlgo::GetCNodeName(node) != kPullOpName) {
            MS_LOG(EXCEPTION) << "The edge between Push and Pull node is invalid.";
          }
          return node;
        }
      }
    }
  }
  return nullptr;
}

GraphId SessionBasic::CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  return executor_->CompileGraph(shared_from_this(), segment, outputs);
}

GraphId SessionBasic::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  MS_EXCEPTION_IF_NULL(executor_);
  return executor_->CompileGraph(shared_from_this(), func_graph);
}

void SessionBasic::BuildGraph(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->BuildGraph(shared_from_this(), graph_id);
}

void SessionBasic::RunOp(const BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  MS_EXCEPTION_IF_NULL(op_run_info);
  executor_->RunOp(shared_from_this(), op_run_info, op_run_info->base_op_run_info.graph_info,
                   &op_run_info->base_op_run_info.input_tensor, outputs, op_run_info->base_op_run_info.input_mask);
}

void SessionBasic::RunOpsInGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                 VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunOpsInGraph(shared_from_this(), graph_id, inputs, outputs);
}

void SessionBasic::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunGraph(shared_from_this(), graph_id, inputs, outputs);
}

void SessionBasic::RunGraphAsync(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                 VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunGraphAsync(shared_from_this(), graph_id, inputs, outputs);
}

void SessionBasic::RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: start run graph. graph id: " << graph_id;
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if none of child graph and no anf output exists
  if (!kernel_graph->executable()) {
    MS_LOG(INFO) << "No child graph has anf output";
    return;
  }
  PreExecuteGraph(kernel_graph, inputs, outputs);
  ExecuteGraph(kernel_graph);
  PostExecuteGraph(kernel_graph, inputs, outputs);
  MS_LOG(INFO) << "Status record: end run graph. graph id: " << graph_id;
}

void SessionBasic::ProcessInputTensorsForHeterogeneous(const std::string &cur_target,
                                                       const std::vector<tensor::TensorPtr> &input_tensors) const {
  for (auto &tensor : input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (device_address != nullptr) {
      if (device_address->GetDeviceType() != device::GetDeviceTypeByName(cur_target)) {
        tensor->data_sync();
        tensor->set_device_address(nullptr);
      }
    }
  }
}

void SessionBasic::RunOpsInGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                     VectorRef *outputs) {
  MS_LOG(INFO) << "Start!";
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::map<AnfNodePtr, size_t> parameter_index;
  GetParameterIndex(kernel_graph.get(), inputs, &parameter_index);
  GraphOutputInfo graph_output_info;
  graph_output_info.graph_outputs = outputs;
  CreateOutputPlaceholder(kernel_graph, inputs, graph_output_info.graph_outputs, &graph_output_info.output_indexes);
  std::map<KernelWithIndex, size_t> cnode_refcount;
  std::map<std::string, size_t> forward_op_output_tensor_id;
  GetRefCount(kernel_graph.get(), &cnode_refcount);
  GetForwardOpOutputRefCount(kernel_graph.get(), inputs, &forward_op_output_tensor_id, parameter_index);
  BuildOpsInGraph(graph_id, parameter_index, inputs, cnode_refcount);

  std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;
  for (const auto &kernel : kernel_graph->execution_order()) {
    // Generate input tensors, tensor masks and input kernel with index
    InputTensorInfo input_tensor_info;
    GetOpInputTensors(kernel, op_output_map, parameter_index, inputs, &input_tensor_info);

    VectorRef op_outputs;
    GraphInfo graph_info;
    // Get OpRunInfo and GraphInfo
    BackendOpRunInfoPtr run_info = GetSingleOpRunInfo(kernel, graph_info, input_tensor_info, &graph_output_info);
    MS_EXCEPTION_IF_NULL(run_info);
    GetSingleOpGraphInfo(kernel, input_tensor_info, &graph_info, run_info);
    run_info->base_op_run_info.graph_info = graph_info;

    // Build and run current single op
    RunOpImplOrigin(graph_info, run_info, &input_tensor_info.input_tensors, &op_outputs,
                    input_tensor_info.input_tensors_mask);
    graph_output_info.graph_output_tensors.clear();
    // Handle inputs and outputs of current op
    ReleaseForwardOpOutput(input_tensor_info.input_tensors, &forward_op_output_tensor_id);
    HandleOpInputs(input_tensor_info.input_kernel, &cnode_refcount, &op_output_map);
    HandleOpOutputs(kernel, op_outputs, cnode_refcount, &op_output_map, &graph_output_info);
  }

  MS_LOG(INFO) << "Finish!";
}

void SessionBasic::EraseValueNodeTensor(const std::vector<int64_t> &tensors_mask,
                                        std::vector<tensor::TensorPtr> *input_tensors) const {
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (input_tensors->size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors->size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  std::vector<tensor::TensorPtr> new_input_tensors;
  for (size_t index = 0; index < tensors_mask.size(); ++index) {
    if (tensors_mask[index] != kValueNodeTensorMask) {
      new_input_tensors.emplace_back(input_tensors->at(index));
    }
  }
  *input_tensors = new_input_tensors;
}

bool SessionBasic::IsGetNextGraph(const std::shared_ptr<KernelGraph> &kernel_graph, std::string *channel_name) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kGetNextOpName) {
      auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
      MS_EXCEPTION_IF_NULL(prim);
      *channel_name = GetValue<std::string>(prim->GetAttr("shared_name"));
      return true;
    }
  }
  return false;
}

void SessionBasic::RunOpRemoveNopNode(const KernelGraphPtr &kernel_graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    opt::RemoveNopNode(kernel_graph.get());
  }
}

void SessionBasic::RunOpHideNopNode(const KernelGraphPtr &kernel_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    opt::HideNopNode(kernel_graph.get());
  }
}

std::vector<uint32_t> SessionBasic::GetAllReduceSplitIndex() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string group = GetCommWorldGroup();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  // PyNative not support multi group allreduce
  group += "sum1";
  return parallel_context->GetAllReduceFusionSplitIndices(group);
}

uint32_t GetBpropGraphGradsCount(const KernelGraphPtr &graph) {
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  MS_LOG(DEBUG) << "Get total graph output size:" << outputs.size();
  // The type of output is CNode or ValueNode.
  // There is no need to calculate grad if the type of output is not CNode.
  return static_cast<uint32_t>(std::count_if(outputs.begin(), outputs.end(), [](const AnfNodePtr &output) {
    return output != nullptr && output->isa<CNode>();
  }));
}

void SetGraphBpropAttr(const KernelGraphPtr &graph) {
  auto &execution_orders = graph->execution_order();
  if (std::any_of(execution_orders.begin(), execution_orders.end(),
                  [](const AnfNodePtr &node) { return node->scope()->name().rfind("Gradient", 0) == 0; })) {
    graph->set_flag(kFlagIsPynativeBpropGraph, true);
    MS_LOG(INFO) << "Match bprop graph";
  }
}

void CheckSplitIndexValid(const vector<uint32_t> &split_index) {
  uint32_t last = 0;
  for (size_t i = 0; i < split_index.size(); ++i) {
    if (split_index[i] <= last && i != 0) {
      MS_LOG(EXCEPTION) << "Invalid split index:" << split_index;
    }
    last = split_index[i];
  }
}

void PreProcessOnSplitIndex(const KernelGraphPtr &graph, vector<uint32_t> *split_index) {
  MS_EXCEPTION_IF_NULL(split_index);
  if (split_index->empty()) {
    return;
  }

  CheckSplitIndexValid(*split_index);
  // calculate split index num
  auto split_index_num = split_index->back();
  // obtain graph output tensor num
  auto grads_count = GetBpropGraphGradsCount(graph);
  if (split_index_num >= grads_count) {
    MS_LOG(WARNING) << "The context configuration all_reduce_fusion_config's upper boundary value should be smaller "
                    << "than total grads count: " << grads_count << ", but got: " << *split_index
                    << ". Now all AllReduce operators will be fused into one AllReduce operator.";
    split_index->clear();
    split_index->push_back(grads_count - 1);
  } else if (split_index_num < grads_count - 1) {
    split_index->push_back(grads_count - 1);
  }
}

void SessionBasic::FinalOptimize(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Start FinalOptimize for graph: " << graph->graph_id();
  opt::CommonFinalOptimization(graph);
  MS_LOG(INFO) << "End FinalOptimize for graph: " << graph->graph_id();
}

void SessionBasic::DumpGraphs(const std::vector<KernelGraphPtr> &graphs) const {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  if (!save_graphs && !json_parser.e2e_dump_enabled() && !json_parser.async_dump_enabled() &&
      !mindspore::RecorderManager::Instance().RdrEnable()) {
    return;
  }
  for (auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    std::string name = "graph_build." + std::to_string(graph->graph_id());
    DumpGraphParams dump_params = {true, static_cast<int>(kWholeStack)};
    (void)mindspore::RDR::RecordAnfGraph(SUBMODULE_ID, name, graph, dump_params, ".ir;.pb");

    auto &kernels = graph->execution_order();
    std::string exec_order_name = "graph_exec_order." + std::to_string(graph->graph_id());
    (void)mindspore::RDR::RecordGraphExecOrder(SUBMODULE_ID, exec_order_name, kernels);
    if (save_graphs) {
      std::string file_name = "graph_build_" + std::to_string(graph->graph_id()) + ".ir";
      DumpIR(file_name, graph, true, kWholeStack);
      DumpIRProto(graph, "vm_build_" + std::to_string(graph->graph_id()));
      DumpIR("trace_code_graph", graph, true, kWholeStack);
    }
    std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_target != kAscendDevice) {
      // Here dump data only with Ascend.
      continue;
    }
    // If the new runtime is used, get rank_id from context via GetRankID(), else get rank_id from rank_id_.
    uint32_t rank_id = rank_id_;
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      const auto &device_context =
        device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id});
      auto deprecated_kernel_executor =
        dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
      if (deprecated_kernel_executor != nullptr) {
        rank_id = deprecated_kernel_executor->GetRankID();
      }
    }
    std::string final_graph = "trace_code_graph_" + std::to_string(graph->graph_id());
    if (json_parser.e2e_dump_enabled() || json_parser.async_dump_enabled()) {
      std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id);
      MS_LOG(INFO) << "Dump graph and exeorder for graph: " << graph->graph_id()
                   << "root_graph_id: " << graph->root_graph_id();
      std::string target_dir = root_dir + "/graphs";
      std::string cst_file_dir = GenerateDumpPath(graph->root_graph_id(), rank_id, true);
      std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";
      DumpIRProtoWithSrcInfo(graph, final_graph, target_dir, kDebugWholeStack);
      if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
        // Dump constant data for old runtime ascend.
        DumpConstantInfo(graph, cst_file_dir);
      }
      DumpIR("trace_code_graph", graph, true, kWholeStack, ir_file_path);
      DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv", root_dir,
                        graph->execution_order());
    }
  }
#endif
}
}  // namespace session
void DumpGraphExeOrder(const std::string &file_name, const std::string &target_dir,
                       const std::vector<CNodePtr> &execution_order) {
  std::string file_path = target_dir + "/execution_order/" + file_name;
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Failed to get real path: [" << file_path << "] in dump graph execution order.";
    return;
  }
  file_path = realpath.value();

  ChangeFileMode(file_path, S_IWUSR);
  // write to csv file
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Failed to open file [" << file_path
                  << "] in dump graph execution order, please check the file access permission and whether disk space "
                     "is available.";
    return;
  }
  ofs << "NodeExecutionOrder-FullNameWithScope\n";
  for (const CNodePtr &node : execution_order) {
    ofs << node->fullname_with_scope() << "\n";
  }
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(file_path, S_IRUSR);
}

uint32_t GetRankId() {
  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = kHcclWorldGroup;
  } else if (backend == kGPUDevice) {
    world_group = kNcclWorldGroup;
  } else {
    MS_LOG(ERROR) << "Invalid backend: " << backend;
    return rank_id;
  }
  if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
    MS_LOG(INFO) << "Failed to get rank id.";
  }
  return rank_id;
}
}  // namespace mindspore
