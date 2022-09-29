
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
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/executor_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/optimizer/helper.h"
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
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/constants.h"
#include "ps/util.h"
#include "ps/ps_context.h"
#include "abstract/abstract_value.h"
#endif
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#else
#include "debug/debugger/proto_exporter_stub.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_exec_order_recorder.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/graph_recorder.h"
#include "runtime/hardware/device_context_manager.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/e2e_dump.h"
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

BaseRef CreateNodeOutputTensors(const AnfNodePtr &anf, const KernelGraphPtr &graph,
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
  size_t size = common::AnfAlgo::GetOutputTensorNum(item_with_index.first);
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
  size_t size = common::AnfAlgo::GetOutputTensorNum(item_with_index.first);
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
    if (tensor_shape[i] < 0 || tensor_shape[i] != input_shape[i]) {
      MS_LOG(EXCEPTION) << "The input tensor's shape: " << tensor_shape
                        << " is not equal to expected shape: " << input_shape << " for input[" << input_index
                        << "] of kernel: " << common::AnfAlgo::GetCNodeName(kernel) << trace::DumpSourceLines(kernel);
    }
  }
}

// 1. Convert the node to make_tuple if the node is a ValueNode<ValueTuple> and it's the input of 'return' node.
// 2. Set the return of graph if node is "Return" node.
void SetReturnNode(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
    constexpr auto kReturnInputIdx = 1;
    auto return_node = node->cast<CNodePtr>();
    graph->set_return(return_node);
    auto graph_output = return_node->input(kReturnInputIdx);
    MS_EXCEPTION_IF_NULL(graph_output);

    // If return's input is value node, then the graph has no kernel, and the pass 'trans tuple to make_tuple' cannot
    // match this pattern because that pass begin with output node but return node. So we add transform value tuple
    // to make_tuple here.
    if (common::AnfAlgo::IsTupleOutput(graph_output) && graph_output->isa<ValueNode>()) {
      return_node->set_input(kReturnInputIdx, graph->TransTupleToMakeTuple(graph_output));
    }
  }
}

void IterateFindTensor(std::vector<ValuePtr> *msTensors, const VectorRef &ref_list) {
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
      msTensors->emplace_back(csr_tensor);
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
    msTensors.emplace_back(tensor_ptr);
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
  }
  return msTensors;
}
}  // namespace

void SessionBasic::InitExecutor(const std::string &device_name, uint32_t device_id) {
  device_id_ = device_id;
  context_ = std::make_shared<Context>(device_name, device_id);
  executor_ = ExecutorManager::Instance().GetExecutor(device_name, device_id);
}

CNodePtr SessionBasic::CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (common::AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto backend_node = graph->GetBackendAnfByFrontAnf(node_input);
    return backend_node->cast<CNodePtr>();
  } else if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  } else {
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto parameter = CreateNewParameterFromCNode(cnode, kernel_graph.get());
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(cnode->abstract());
    auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()));
    auto return_node = kernel_graph->NewCNode({primitive, parameter});
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph));
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  }
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

std::vector<AnfNodePtr> SessionBasic::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  if (cnode->inputs().size() <= 1) {
    cnode_inputs = switch_cnode->inputs();
    return cnode_inputs;
  }
  std::vector<AnfNodePtr> switch_inputs = {switch_cnode->input(kAnfPrimitiveIndex),
                                           switch_cnode->input(kFirstDataInputIndex)};
  for (size_t index = kSwitchTrueBranchIndex; index < switch_cnode->inputs().size(); index++) {
    auto node = switch_cnode->input(index);
    // there is real input in call, should put it to true and false branch in switch
    if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      auto partial_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      std::vector<AnfNodePtr> partial_inputs = partial_node->inputs();
      // Put all call args at the end of partial inputs.
      for (size_t i = kFirstDataInputIndex; i < cnode->size(); ++i) {
        (void)partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(i)));
      }
      auto new_partial = graph->NewCNode(partial_inputs);
      (void)switch_inputs.emplace_back(new_partial);
    }
  }
  if (switch_inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch inputs size: " << switch_inputs.size() << "less than " << kSwitchInputSize;
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  (void)cnode_inputs.emplace_back(switch_node);
  return cnode_inputs;
}

void SessionBasic::ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph,
                                      const std::vector<AnfNodePtr> &real_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  // func1 =switch(branch1, branch2)
  // func2 = func1(param1)
  // out = func2(param2)
  // process the last cnode(func2), not func1 which abstract is AbstractFunction
  if (cnode->abstract()->isa<abstract::AbstractFunction>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto return_input = ret->input(kFirstDataInputIndex);
  // return node is a function
  std::vector<AnfNodePtr> call_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial)) {
    auto return_input_cnode = return_input->cast<CNodePtr>();
    auto partial_inputs = return_input_cnode->inputs();
    (void)call_inputs.insert(call_inputs.cend(), partial_inputs.cbegin() + kFirstDataInputIndex, partial_inputs.cend());
  } else if (IsValueNode<KernelGraph>(return_input)) {  // return node is kernel graph
    call_inputs.emplace_back(return_input);
  } else {  // return node is value node
    KernelGraphPtr kernel_graph = NewKernelGraph();
    auto valid_inputs = kernel_graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = kernel_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    std::vector<AnfNodePtr> cnode_inputs = {return_input};
    for (auto &real_input : real_inputs) {
      auto new_parameter = kernel_graph->NewParameter(real_input->abstract());
      valid_inputs->push_back(true);
      graph_inputs->push_back(new_parameter);
      cnode_inputs.push_back(new_parameter);
    }
    auto new_cnode = kernel_graph->NewCNode(cnode_inputs);
    new_cnode->set_abstract(cnode->abstract());
    std::vector<AnfNodePtr> return_inputs = {
      kernel_graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()))), new_cnode};
    auto return_node = kernel_graph->NewCNode(return_inputs);
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    call_inputs.push_back(std::make_shared<ValueNode>(kernel_graph));
  }

  // new call node inputs
  for (auto &input_node : real_inputs) {
    auto parameter_for_input = CreateNewParameterFromCNode(input_node, graph);
    call_inputs.emplace_back(parameter_for_input);
  }

  auto call_node = graph->NewCNode(call_inputs);
  call_node->set_abstract(cnode->abstract());
  // update return input
  ret->set_input(kFirstDataInputIndex, call_node);
}

std::vector<AnfNodePtr> SessionBasic::CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_layer_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_layer_cnode);
  std::vector<AnfNodePtr> switch_layer_inputs = {switch_layer_cnode->input(kAnfPrimitiveIndex),
                                                 switch_layer_cnode->input(kFirstDataInputIndex)};
  auto make_tuple_node = switch_layer_cnode->input(kSwitchLayerBranchesIndex);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  auto node = make_tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto make_tuple_inputs = node->inputs();
  // there are real inputs in call, should put it to make_tuple in switch_layer
  std::vector<AnfNodePtr> real_inputs;
  for (size_t idx = kFirstDataInputIndex; idx < cnode->inputs().size(); ++idx) {
    real_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(idx)));
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())))};
  for (size_t idx = kFirstDataInputIndex; idx < make_tuple_inputs.size(); idx++) {
    auto partial_idx = make_tuple_inputs[idx];
    MS_EXCEPTION_IF_NULL(cnode->abstract());
    std::vector<AnfNodePtr> new_partial_inputs;
    KernelGraphPtr partial_kernel_graph;
    // switch_layer node input is partial cnode
    if (common::AnfAlgo::CheckPrimitiveType(partial_idx, prim::kPrimPartial)) {
      auto partial_node = partial_idx->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      auto partial_input = partial_node->input(kFirstDataInputIndex);
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_input);
      new_partial_inputs = partial_node->inputs();
    } else if (IsValueNode<KernelGraph>(partial_idx)) {  // switch_layer node input is kernel graph value node
      new_partial_inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name())));
      new_partial_inputs.emplace_back(partial_idx);
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_idx);
    }
    // when branch in swich_layer return function
    MS_EXCEPTION_IF_NULL(partial_kernel_graph);
    auto ret = partial_kernel_graph->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    auto return_input = ret->input(kFirstDataInputIndex);
    if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial) || return_input->isa<ValueNode>()) {
      ProcessNodeRetFunc(cnode, partial_kernel_graph.get(), real_inputs);
    }
    // partial node add input args
    (void)new_partial_inputs.insert(new_partial_inputs.cend(), real_inputs.cbegin(), real_inputs.cend());
    // create new partial node
    auto new_partial = graph->NewCNode(new_partial_inputs);
    new_make_tuple_inputs.emplace_back(new_partial);
  }
  auto new_make_tuple = graph->NewCNode(new_make_tuple_inputs);
  auto abstract = make_tuple_node->abstract();
  if (abstract == nullptr) {
    abstract = std::make_shared<abstract::AbstractTuple>(AbstractBasePtrList());
  }
  new_make_tuple->set_abstract(abstract);
  switch_layer_inputs.emplace_back(new_make_tuple);
  auto new_switch_layer = graph->NewCNode(switch_layer_inputs);
  cnode_inputs.emplace_back(new_switch_layer);
  return cnode_inputs;
}

std::vector<AnfNodePtr> SessionBasic::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  // create primitive of cnode:call(partial or switch or switch_layer)
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  if (cnode_input == nullptr) {
    MS_LOG(ERROR) << "CNode input[0] is CNode:" << attr_input->DebugString() << ", but input[0] has not been created.";
    return {};
  }
  // if the node is partial, insert the inputs of partial to the call
  if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    (void)std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                         std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                           MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                           return graph->GetBackendAnfByFrontAnf(node);
                         });
    return cnode_inputs;
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    return CreateCallSwitchInputs(cnode, graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitchLayer)) {
    return CreateCallSwitchLayerInputs(cnode, graph);
  }
  MS_LOG(ERROR) << "CNode:" << cnode->DebugString() << " input[0]" << cnode_input->DebugString()
                << "must be partial or switch or switch_layer.";
  return {};
}

std::vector<AnfNodePtr> SessionBasic::CreateValueNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (common::AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  } else {
    // create primitive of cnode:call
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
    // create a ValueNode<KernelGraph> as input of cnode:call
    if (graph->GetBackendAnfByFrontAnf(attr_input) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(attr_input));
    } else {
      auto new_value_node = CreateValueNodeKernelGraph(attr_input, graph);
      if (new_value_node != nullptr) {
        cnode_inputs.emplace_back(new_value_node);
      }
    }
  }
  return cnode_inputs;
}

void SessionBasic::CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
    for (size_t index = kSwitchTrueBranchIndex; index < cnode->inputs().size(); index++) {
      auto node_input = cnode->input(index);
      auto switch_input = CreateSwitchInput(cnode, node_input, graph);
      (void)cnode_inputs->emplace_back(switch_input);
    }
  } else {
    for (size_t input_idx = kFirstDataInputIndex; input_idx < cnode->inputs().size(); input_idx++) {
      auto anf = cnode->input(input_idx);
      MS_EXCEPTION_IF_NULL(anf);
      // anf has been created before
      if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
        (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
        continue;
      } else if (IsValueNode<None>(anf)) {
        continue;
      }
      MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
    }
  }
}

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (IsValueNode<FuncGraph>(attr_input)) {
    // cnode is a graph or a call
    cnode_inputs = CreateValueNode(cnode, graph);
  } else if (attr_input->isa<CNode>()) {
    // cnode ia a call (partial/switch/switch_layer)
    // 1. take the args of call to the partial node, as the real_args to call switch's or switch_layer's child graph
    // 2. the call in frontend is map to the partial/switch/switch_layer in backend and haven't been created
    cnode_inputs = CreateSwitchOrPartialNode(cnode, graph);
    if (cnode_inputs.empty()) {
      MS_LOG_ERROR << "Create switch or partial failed, cnode:" << cnode->DebugString();
      return nullptr;
    }
  } else {
    // get primitive of old node
    auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }
  // handle inputs of cnode except primitive
  CreateCNodeInputs(cnode, graph, &cnode_inputs);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNodeWithInfos(cnode_inputs, cnode);
  // if the cnode is call switch, remove call
  if (new_cnode->inputs().size() > 1) {
    auto first_input = new_cnode->input(kFirstDataInputIndex);
    MS_EXCEPTION_IF_NULL(first_input);
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitch)) {
      new_cnode = first_input->cast<CNodePtr>();
    }
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitchLayer)) {
      auto abstract = cnode->abstract();
      new_cnode = first_input->cast<CNodePtr>();
      new_cnode->set_abstract(abstract);
    }
  }
  return new_cnode;
}

ValueNodePtr SessionBasic::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto sub_func_graph = common::AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph.get()) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph.get()];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr SessionBasic::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (param_value != nullptr) {
    new_parameter = param_value->parameter();
    if (new_parameter == nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
      new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
      param_value->set_parameter(new_parameter);
    }
  } else {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  }

  new_parameter->IncreaseUsedGraphCount();

  return new_parameter;
}

void SessionBasic::GetSingleOpGraphInfo(const CNodePtr &kernel, const InputTensorInfo &tensor_info,
                                        GraphInfo *graph_info) const {
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
  buf << GetOpRunDeviceTarget(prim) << "_";
  buf << prim->id();
  bool has_const_input = false;
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto &tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->base_shape_ptr() != nullptr) {
      buf << tensor->base_shape_ptr()->ToString();
    } else {
      buf << tensor->shape();
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
      has_const_input = true;
      buf << common::AnfAlgo::GetTensorValueString(tensor);
    }
    buf << "_";
  }

  // Get attr info
  const auto &attr_map = prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&buf](const auto &element) { buf << element.second->ToString(); });

  // Generally, different inputs can have different output; but different constant inputs may lead to different output
  if (has_const_input) {
    buf << "_";
    const AbstractBasePtr &abstract = kernel->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    auto build_shape = abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(build_shape);
    auto build_type = abstract->BuildType();
    MS_EXCEPTION_IF_NULL(build_type);
    // Get output shape
    buf << build_shape->ToString();
    // Get output dtype
    buf << build_type->type_id();
  }
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

  bool is_gradient_out =
    graph_output_info != nullptr &&
    std::any_of(graph_output_info->output_indexes.cbegin(), graph_output_info->output_indexes.cend(),
                [cnode](const std::pair<KernelWithIndex, std::vector<std::vector<size_t>>> &output_index) {
                  return output_index.first.first == cnode;
                });
  pynative::BaseOpRunInfo base_op_run_info;
  base_op_run_info.has_dynamic_input = common::AnfAlgo::IsNodeInputDynamicShape(cnode);
  base_op_run_info.has_dynamic_output = shape->IsDynamic();
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
  return std::make_shared<BackendOpRunInfo>(base_op_run_info, primitive.get(), false, is_gradient_out);
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
                                              std::map<std::string, size_t> *forward_op_output_tensor_id) const {
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
      }
    }
  }
  // Forward op output use as sens, so need add reference
  for (const auto &tensor : inputs) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_forward_output()) {
      (*forward_op_output_tensor_id)[tensor->id()] += 1;
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
  auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kTBE, is_dynamic_shape);
  if (op_info_ptr == nullptr) {
    return;
  }
  auto inputs_ptr = op_info_ptr->inputs_ptr();
  for (size_t i = 0; i < inputs_ptr.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs_ptr[i]);
    if (!inputs_ptr[i]->value_depend().empty() && inputs_ptr[i]->value_depend() != "ignored") {
      const_input_attr_index->push_back(i + 1);
    }
  }
}

void SessionBasic::GetOpInputTensors(const CNodePtr &cnode,
                                     const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                                     const std::map<AnfNodePtr, size_t> &parameter_index,
                                     const std::vector<tensor::TensorPtr> &graph_inputs,
                                     InputTensorInfo *input_tensor_info) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_tensor_info);
  std::vector<size_t> const_input_attr_index = {};
  GetConstValueDepend(cnode, &const_input_attr_index);
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
        is_value_node =
          std::find(const_input_attr_index.begin(), const_input_attr_index.end(), i) != const_input_attr_index.end();
      }
      input_tensor_info->input_tensors_mask.emplace_back(is_value_node ? kValueNodeTensorMask
                                                                       : kParameterDataTensorMask);
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

bool SessionBasic::CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // create a new cnode object
  auto new_cnode = CreateNewCNode(cnode, graph);
  if (new_cnode == nullptr) {
    return false;
  }
  new_cnode->set_abstract(cnode->abstract());
  std::string fullname;
  if (cnode->input(kAnfPrimitiveIndex)->isa<CNode>()) {
    fullname = cnode->input(kAnfPrimitiveIndex)->fullname_with_scope();
  } else if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
    fullname = cnode->input(kFirstDataInputIndex)->fullname_with_scope();
  } else {
    fullname = cnode->fullname_with_scope();
  }
  new_cnode->set_fullname_with_scope(fullname);
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendMapAdd(node, new_cnode);
  SetReturnNode(new_cnode, graph);
  return true;
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                std::vector<KernelGraphPtr> *all_out_graph,
                                                                DeviceType device_target) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph.get()] = graph;
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  graph->set_device_target(device_target);
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    // Create parameter
    if (node->isa<Parameter>()) {
      auto graph_inputs = graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(graph_inputs);
      auto new_parameter = CreateNewParameter(node, graph.get());
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendMapAdd(node, new_parameter);
      continue;
    }
    // Create value node
    if (node->isa<ValueNode>()) {
      // Create common value node
      if (!IsValueNode<FuncGraph>(node)) {
        (void)CreateNewValueNode(node, graph.get());
        continue;
      }
      // Create child kernel graph according ValueNode<FuncGraph>
      FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(node);
      if (front_backend_graph_map_.find(child_graph.get()) == front_backend_graph_map_.end()) {
        (void)ConstructKernelGraph(child_graph, all_out_graph, device_target);
      }
      (void)CreateValueNodeKernelGraph(node, graph.get());
      continue;
    }
    // Create cnode
    if (!CreateCNodeOfKernelGraph(node, graph.get())) {
#ifdef ENABLE_DUMP_IR
      DumpIR("construct_kernel_graph_fail.ir", func_graph);
#endif
      MS_LOG(EXCEPTION) << "Construct func graph " << func_graph->ToString() << " failed."
                        << trace::DumpSourceLines(node);
    }
  }

  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  FuncGraphManagerPtr manager = MakeManager({graph});
  graph->SetInputNodes();
  SetInputNodeUsage(graph, manager);
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif

  all_out_graph->push_back(graph);
  return graph;
}

void SessionBasic::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto backend_parameter = graph->GetBackendAnfByFrontAnf(parameter);
    if (backend_parameter == nullptr) {
      // for example "def f(x,y,z) {return x + y}", parameter z in unused
      auto new_parameter = CreateNewParameter(parameter, graph);
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendMapAdd(parameter, new_parameter);
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    graph_inputs->push_back(backend_parameter);
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
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  MS_EXCEPTION_IF_NULL(graph);
  if (common::AnfAlgo::GetOutputTensorNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < common::AnfAlgo::GetOutputTensorNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToLong(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int64Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
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
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info->base_op_run_info.abstract);
  // get output dynamic shape info
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(op_run_info->base_op_run_info.has_dynamic_input),
                               cnode);
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
  if (is_ascend) {
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
  GetForwardOpOutputRefCount(kernel_graph.get(), inputs, &forward_op_output_tensor_id);
  BuildOpsInGraph(graph_id, parameter_index, inputs, cnode_refcount);

  std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;
  for (const auto &kernel : kernel_graph->execution_order()) {
    // Generate input tensors, tensor masks and input kernel with index
    InputTensorInfo input_tensor_info;
    GetOpInputTensors(kernel, op_output_map, parameter_index, inputs, &input_tensor_info);

    VectorRef op_outputs;
    // Get OpRunInfo and GraphInfo
    GraphInfo graph_info;
    GetSingleOpGraphInfo(kernel, input_tensor_info, &graph_info);
    BackendOpRunInfoPtr run_info = GetSingleOpRunInfo(kernel, graph_info, input_tensor_info, &graph_output_info);

    // Build and run current single op
    RunOpImplOrigin(graph_info, run_info, &input_tensor_info.input_tensors, &op_outputs,
                    input_tensor_info.input_tensors_mask);
    graph_output_info.graph_output_tensors.clear();
    // Handle inputs and outputs of current op
    ReleaseForwardOpOutput(input_tensor_info.input_tensors, &forward_op_output_tensor_id);
    HandleOpInputs(input_tensor_info.input_kernel, &cnode_refcount, &op_output_map);
    HandleOpOutputs(kernel, op_outputs, cnode_refcount, &op_output_map, &graph_output_info);
    // Save grad node to Bucket
    if (kernel_graph->has_flag(kFlagIsPynativeBpropGraph)) {
      AddGradAddrToBucket(graph_id, graph_output_info.graph_output_tensors);
    }
  }
  // Clear bucket resources every step
  if (kernel_graph->has_flag(kFlagIsPynativeBpropGraph)) {
    ClearAllBucket(graph_id);
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

std::vector<uint32_t> GenerateBucketSizeList(const KernelGraphPtr &graph, const std::vector<uint32_t> &split_index) {
  if (split_index.empty()) {
    auto grads_count = GetBpropGraphGradsCount(graph);
    MS_LOG(DEBUG) << "Get valid grads size:" << grads_count;
    if (grads_count == 0) {
      MS_LOG(EXCEPTION) << "Bprop graph has no grad";
    }
    return {grads_count};
  }

  std::vector<uint32_t> bucket_size_list;
  uint32_t old_index = 0;
  for (const auto &index : split_index) {
    if (old_index == 0) {
      bucket_size_list.emplace_back(index - old_index + 1);
    } else {
      bucket_size_list.emplace_back(index - old_index);
    }
    old_index = index;
  }
  return bucket_size_list;
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

void SessionBasic::InitAllBucket(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start init all bucket. graph id: " << graph->graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (!pynative_mode || (parallel_mode != parallel::kDataParallel && parallel_mode != parallel::kSemiAutoParallel &&
                         parallel_mode != parallel::kAutoParallel)) {
    return;
  }
  SetGraphBpropAttr(graph);

  if (!graph->has_flag(kFlagIsPynativeBpropGraph)) {
    return;
  }

  std::vector<std::shared_ptr<device::Bucket>> bucket_list;
  // Create bucket for every split allreduce ops
  auto split_index = GetAllReduceSplitIndex();
  PreProcessOnSplitIndex(graph, &split_index);
  auto bucket_size_list = GenerateBucketSizeList(graph, split_index);
  uint32_t bucket_id = 0;
  for (const auto &bucket_size : bucket_size_list) {
    MS_LOG(INFO) << "Create new bucket:" << bucket_id << " size:" << bucket_size;
    std::shared_ptr<device::Bucket> bucket = nullptr;
    if (device_context != nullptr) {
      auto deprecated_kernel_executor =
        dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
      if (deprecated_kernel_executor != nullptr) {
        bucket = deprecated_kernel_executor->CreateBucket(bucket_id++, bucket_size);
      } else {
        MS_LOG(EXCEPTION) << "Not Support CreateBucket() in Device Context.";
      }
    } else {
      bucket = CreateBucket(bucket_id++, bucket_size);
    }
    bucket_list.emplace_back(bucket);
  }

  auto bucket_ret = bucket_map_.try_emplace(graph->graph_id(), bucket_list);
  if (!bucket_ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate bucket_map_ graph key:" << graph->graph_id();
  }
  // set all free bucket index to 0
  auto free_bucket_ret = free_bucket_id_map_.try_emplace(graph->graph_id(), 0);
  if (!free_bucket_ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate free_bucket_id_map_ graph key:" << graph->graph_id();
  }
  MS_LOG(INFO) << "Status record: end init all bucket. graph id: " << graph->graph_id();
}

void SessionBasic::DoAllReduceOnGrads(const std::string &actor_info, const std::vector<tensor::TensorPtr> &outputs,
                                      const device::DeviceContext *device_context) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kDataParallel && parallel_mode != parallel::kSemiAutoParallel &&
      parallel_mode != parallel::kAutoParallel) {
    MS_LOG(DEBUG) << "No need to do AllReduce";
    return;
  }

  MS_EXCEPTION_IF_NULL(device_context);
  std::shared_ptr<device::Bucket> bucket;
  auto iter = actor_set_to_bucket_.find(actor_info);
  if (iter == actor_set_to_bucket_.end()) {
    auto deprecated_kernel_executor =
      dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
    if (deprecated_kernel_executor != nullptr) {
      static size_t bucket_id = 0;
      bucket = deprecated_kernel_executor->CreateBucket(SizeToUint(bucket_id++), SizeToUint(outputs.size()));
    } else {
      MS_LOG(EXCEPTION) << "Not Support CreateBucket() in Device Context.";
    }
    actor_set_to_bucket_[actor_info] = bucket;
  } else {
    bucket = iter->second;
  }

  MS_EXCEPTION_IF_NULL(bucket);
  for (auto &output : outputs) {
    bucket->AddGradTensor(output);
  }
  if (bucket->full()) {
    bucket->Launch();
  } else {
    MS_LOG(EXCEPTION) << "Do AllReduce for " << actor_info << " failed, grad size " << outputs.size() << " bucket size "
                      << bucket->bucket_size() << " not equal";
  }
  bucket->Release();
}

void SessionBasic::AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kDataParallel && parallel_mode != parallel::kAutoParallel &&
      parallel_mode != parallel::kSemiAutoParallel) {
    return;
  }

  auto iter = bucket_map_.find(graph_id);
  if (iter == bucket_map_.end()) {
    MS_LOG(EXCEPTION) << "unknown graph id:" << graph_id;
  }
  auto &bucket_list = iter->second;
  auto free_bucket_iter = free_bucket_id_map_.find(graph_id);
  if (free_bucket_iter == free_bucket_id_map_.end()) {
    MS_LOG(EXCEPTION) << "unknown free graph id:" << graph_id;
  }

  auto free_bucket_index = free_bucket_iter->second;
  for (auto &tensor : grad_tensor) {
    if (free_bucket_index >= bucket_list.size()) {
      MS_LOG(EXCEPTION) << "Invalid free bucket id:" << free_bucket_iter->second
                        << " total bucket num:" << bucket_list.size();
    }
    auto &free_bucket = bucket_list[free_bucket_index];
    free_bucket->AddGradTensor(tensor);
    if (free_bucket->full()) {
      // AllReduce need to wait for the kernel execution of bprop to complete.
      runtime::OpExecutor::GetInstance().Wait();
      MS_LOG(INFO) << "bucket is full";
      free_bucket->Launch();
      free_bucket_index = ++free_bucket_iter->second;
      MS_LOG(INFO) << "new free bucket:" << free_bucket_index;
    }
  }
}

void SessionBasic::ClearAllBucket(const GraphId &graph_id) {
  auto iter = bucket_map_.find(graph_id);
  if (iter != bucket_map_.end()) {
    auto bucket_list = iter->second;
    for (auto &bucket : bucket_list) {
      MS_LOG(INFO) << "Clear bucket:" << bucket->id();
      bucket->Release();
    }
  }
  auto free_iter = free_bucket_id_map_.find(graph_id);
  if (free_iter != free_bucket_id_map_.end()) {
    free_iter->second = 0;
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
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
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
      if (graph->is_dynamic_shape()) {
        MS_LOG(EXCEPTION) << "Dump is not supported for dynamic shape!";
      }
      std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id);
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
