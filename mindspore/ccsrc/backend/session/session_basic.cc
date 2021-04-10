/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/session/session_basic.h"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <utility>

#include "ops/primitive_c.h"
#include "ir/manager.h"
#include "abstract/utils.h"
#include "backend/kernel_compiler/common_utils.h"
#include "base/core_ops.h"
#include "base/base_ref_utils.h"
#include "common/trans.h"
#include "utils/config_manager.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/executor_manager.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/ms_utils.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "utils/utils.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "debug/common.h"
#include "utils/trace_base.h"
#include "frontend/parallel/context.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/ps_cache/ps_cache_manager.h"
#include "ps/constants.h"
#include "ps/util.h"
#include "ps/ps_context.h"
#include "abstract/abstract_value.h"
#endif

namespace mindspore {
namespace session {
static std::shared_ptr<std::map<ParamInfoPtr, ParameterPtr>> python_paras;
void ClearPythonParasMap() { python_paras = nullptr; }
namespace {
const int kSummaryGetItem = 2;
const size_t max_depth = 128;
bool IsShapeDynamic(const abstract::ShapePtr &shape) {
  if (shape == nullptr) {
    return false;
  }
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}
bool RecursiveCheck(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, size_t *idx) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::IsRealKernel(node)) {
    return true;
  }
  (*idx) += 1;
  // max recursion depth
  if (*idx <= max_depth) {
    auto users = manager->node_users()[node];
    if (std::any_of(users.begin(), users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
          return RecursiveCheck(manager, kernel.first, idx);
        })) {
      return true;
    }
  }
  return false;
}

bool IsUsedByRealKernel(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  auto node_users = manager->node_users()[node];
  size_t idx = 0;
  if (std::any_of(node_users.begin(), node_users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
        return RecursiveCheck(manager, kernel.first, &idx);
      })) {
    return true;
  }
  return false;
}

bool CheckIfNeedCreateOutputTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>()) {
    auto node_ptr = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(node_ptr);
    if (!node_ptr->is_used_by_real_kernel()) {
      return true;
    }
  }
  return false;
}

ParamInfoPtr GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->param_info();
}

tensor::TensorPtr CreateCNodeOutputTensor(const session::KernelWithIndex &node_output_pair,
                                          const KernelGraphPtr &graph) {
  auto &node = node_output_pair.first;
  auto &output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  TypeId type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (type_id == kTypeUnknown) {
    type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  tensor::TensorPtr tensor = nullptr;
  std::vector<int64_t> temp_shape;
  if (graph->IsUniqueTargetInternalOutput(node, output_index)) {
    temp_shape.emplace_back(1);
    tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
    tensor->set_sync_status(kNoNeedSync);
    tensor->SetNeedWait(true);
    tensor->SetIsGraphOutput();
    return tensor;
  }

  tensor = graph->GetInternalOutputTensor(node, output_index);
  if (tensor == nullptr) {
    auto shape = AnfAlgo::GetOutputInferShape(node, output_index);
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    bool is_internal_output = graph->IsInternalOutput(node, output_index);
    if (is_internal_output) {
      graph->AddInternalOutputTensor(node, output_index, tensor);
    }
  }
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
  // if in pynative mode,data only copied to host when user want to print data
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
    tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
  } else {
    tensor->set_sync_status(kNeedSyncDeviceToHost);
  }
  tensor->SetNeedWait(true);
  tensor->SetIsGraphOutput();
  return tensor;
}

static bool IsPynativeMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
}

BaseRef CreateNodeOutputTensor(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                               const std::vector<tensor::TensorPtr> &input_tensors,
                               std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto &node = node_output_pair.first;
  auto &output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_LOG(INFO) << "Create tensor for output[" << node->DebugString() << "] index[" << node_output_pair.second << "]";
  if (HasAbstractMonad(node)) {
    return std::make_shared<tensor::Tensor>(int64_t(0), kBool);
  }
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  bool output_addr_exist = AnfAlgo::OutputAddrExist(node, output_index);
  if (!output_addr_exist || (CheckIfNeedCreateOutputTensor(node) && !IsPynativeMode())) {
    if (node->isa<Parameter>()) {
      for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
        if (input_idx >= input_tensors.size()) {
          MS_LOG(EXCEPTION) << "Input idx:" << input_idx << "out of range:" << input_tensors.size();
        }
        if (graph->inputs()[input_idx] == node) {
          return input_tensors[input_idx];
        }
      }
      if (!output_addr_exist) {
        MS_LOG(EXCEPTION) << "Parameter : " << node->DebugString() << " has no output addr";
      }
    }
  }
  auto tensor = CreateCNodeOutputTensor(node_output_pair, graph);
  (*tensor_to_node)[tensor] = node_output_pair;
  return tensor;
}

BaseRef CreateNodeOutputTensors(const AnfNodePtr &anf, const KernelGraphPtr &graph,
                                const std::vector<tensor::TensorPtr> &input_tensors,
                                std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_LOG(INFO) << "Create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(INFO) << "Create tensor for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      if (!AnfAlgo::CheckPrimitiveType(cnode->input(i), prim::kPrimControlDepend)) {
        auto out = CreateNodeOutputTensors(cnode->input(i), graph, input_tensors, tensor_to_node);
        ret.push_back(out);
      }
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }
  return CreateNodeOutputTensor(item_with_index, graph, input_tensors, tensor_to_node);
}

ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<None>()) {
    return nullptr;
  }
  auto new_value_node = graph->NewValueNode(value_node);
  graph->FrontBackendlMapAdd(anf, new_value_node);
  graph->AddValueNodeToGraph(new_value_node);
  return new_value_node;
}

size_t LoadCtrlInputTensor(const std::shared_ptr<KernelGraph> &graph, std::vector<tensor::TensorPtr> *inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Load kInputCtrlTensors";
  auto inputs_params = graph->input_ctrl_tensors();
  if (inputs_params == nullptr) {
    return 0;
  }
  if (inputs_params->size() < 3) {
    MS_LOG(EXCEPTION) << "Illegal inputs_params size";
  }
  // update current loop tensor to 0 per iterator
  auto cur_loop_tensor = (*inputs_params)[0];
  MS_EXCEPTION_IF_NULL(cur_loop_tensor);
  auto *cur_val = static_cast<int32_t *>(cur_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(cur_val);
  *cur_val = 0;
  cur_loop_tensor->set_sync_status(kNeedSyncHostToDevice);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(cur_loop_tensor);

  // update next loop tensor to 0 per iterator
  auto next_loop_tensor = (*inputs_params)[1];
  MS_EXCEPTION_IF_NULL(next_loop_tensor);
  auto *next_val = static_cast<int32_t *>(next_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(next_val);
  *next_val = 0;
  next_loop_tensor->set_sync_status(kNeedSyncHostToDevice);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(next_loop_tensor);

  auto epoch_tensor = (*inputs_params)[2];
  MS_EXCEPTION_IF_NULL(epoch_tensor);
  auto *epoch_val = static_cast<int32_t *>(epoch_tensor->data_c());
  MS_EXCEPTION_IF_NULL(epoch_val);
  *epoch_val = graph->current_epoch();
  epoch_tensor->set_sync_status(kNeedSyncHostToDevice);
  inputs->push_back(epoch_tensor);
  MS_LOG(INFO) << "Load epoch_val:" << *epoch_val;

  graph->set_current_epoch(graph->current_epoch() + 1);

  return inputs_params->size();
}

ValueNodePtr ConstructRunOpValueNode(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto value_node = std::make_shared<ValueNode>(input_tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  // construct abstract of value node
  auto type_of_tensor = input_tensor->Dtype();
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
  value_node->set_abstract(abstract);
  // add value node to graph
  auto input_value_node = graph->NewValueNode(value_node);
  graph->AddValueNodeToGraph(input_value_node);
  return input_value_node;
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     int64_t tensor_mask) {
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
  if (device_address == nullptr) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
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
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
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

bool ExistSummaryNode(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = DeepLinkedGraphSearch(ret);
  for (auto &n : all_nodes) {
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      return true;
    }
  }
  return false;
}

bool IgnoreCreateParameterForMakeTuple(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &node_inputs = cnode->inputs();
  for (size_t i = 1; i < node_inputs.size(); ++i) {
    if (!AnfAlgo::CheckPrimitiveType(node_inputs[i], prim::kPrimControlDepend)) {
      return false;
    }
  }
  return true;
}

void GetParameterIndex(const KernelGraph *graph, const std::vector<tensor::TensorPtr> &inputs,
                       std::map<AnfNodePtr, size_t> *parameter_index) {
  size_t index = 0;
  for (const auto &input_node : graph->inputs()) {
    auto params = AnfAlgo::GetAllOutput(input_node);
    for (const auto &param : params) {
      if (index >= inputs.size()) {
        MS_LOG(EXCEPTION) << "Parameter size out of range. Parameter index: " << index
                          << ", input size: " << inputs.size();
      }
      const auto &input = inputs[index];
      // Check shape of input and parameter
      const auto &input_shape = input->shape();
      const auto &param_shape = AnfAlgo::GetOutputInferShape(param, 0);
      if (input_shape.size() != param_shape.size()) {
        MS_LOG(EXCEPTION) << "Shapes of input and parameter are different, input index: " << index
                          << ", parameter: " << param->fullname_with_scope();
      }
      for (size_t i = 0; i < input_shape.size(); i += 1) {
        if (input_shape[i] < 0 || static_cast<size_t>(input_shape[i]) != param_shape[i]) {
          MS_LOG(EXCEPTION) << "Shapes of input and parameter are different, input index: " << index
                            << ", parameter: " << param->fullname_with_scope();
        }
      }
      parameter_index->emplace(param, index++);
    }
  }
}

BaseRef CreateNodeOutputPlaceholder(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                                    const std::vector<tensor::TensorPtr> &input_tensors,
                                    const std::vector<size_t> &indexes,
                                    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  auto &node = node_output_pair.first;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(output_indexes);
  MS_LOG(INFO) << "Create placeholder for output[" << node->DebugString() << "] index[" << node_output_pair.second
               << "]";
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  if (node->isa<Parameter>()) {
    for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
      if (input_idx >= input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Input idx:" << input_idx << "out of range:" << input_tensors.size();
      }
      if (graph->inputs()[input_idx] == node) {
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
  MS_LOG(INFO) << "Create placeholder for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(INFO) << "Create placeholder for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
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

void CreateOutputPlaceholder(const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &input_tensors,
                             VectorRef *outputs,
                             std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_indexes);
  auto anf_outputs = kernel_graph->outputs();
  size_t index = 0;
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output placeholder[" << item->DebugString() << "]";
    std::vector<size_t> indexes{index++};
    outputs->emplace_back(CreateNodeOutputPlaceholder(item, kernel_graph, input_tensors, indexes, output_indexes));
  }
}

void GetRefCount(const KernelGraph *graph, std::map<KernelWithIndex, size_t> *ref_count) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &kernel : graph->execution_order()) {
    for (size_t i = 1; i < kernel->inputs().size(); i += 1) {
      const auto &input = kernel->input(i);
      auto kernel_with_index = AnfAlgo::VisitKernel(input, 0);
      const auto &node = kernel_with_index.first;
      if (node->isa<CNode>()) {
        (*ref_count)[kernel_with_index] += 1;
      }
    }
  }
}

void HandleOpInputs(const std::set<KernelWithIndex> &input_kernel, std::map<KernelWithIndex, size_t> *ref_count,
                    std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) {
  MS_EXCEPTION_IF_NULL(ref_count);
  MS_EXCEPTION_IF_NULL(op_output_map);
  for (auto &kernel_with_index : input_kernel) {
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    if (!kernel_with_index.first->isa<CNode>()) {
      continue;
    }
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

void HandleOpOutputs(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                     const std::map<KernelWithIndex, std::vector<std::vector<size_t>>> &output_indexes,
                     const std::map<KernelWithIndex, size_t> &ref_count,
                     std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map, VectorRef *outputs,
                     std::vector<TensorPtr> *runop_output_tensors) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_output_map);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(runop_output_tensors);
  auto output_tensors = TransformVectorRefToMultiTensor(op_outputs);
  if (output_tensors.size() > op_outputs.size()) {
    MS_LOG(EXCEPTION) << "Op output contains tuple, node = " << kernel->DebugString();
  }
  size_t out_index = 0;
  for (const auto &output_tensor : output_tensors) {
    auto kernel_with_index = make_pair(kernel, out_index++);
    if (ref_count.find(kernel_with_index) != ref_count.end()) {
      (*op_output_map)[kernel_with_index] = output_tensor;
    }
    const auto &iter = output_indexes.find(kernel_with_index);
    if (iter == output_indexes.end()) {
      continue;
    }
    const std::vector<std::vector<size_t>> &multiple_ref_indexes = iter->second;
    for (const auto &ref_indexes : multiple_ref_indexes) {
      size_t n = 0;
      const VectorRef *cur_vector_ref = outputs;
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
      tensor_ref = output_tensor;
      runop_output_tensors->emplace_back(output_tensor);
    }
  }
}
}  // namespace

GraphId SessionBasic::graph_sum_ = 0;

void SessionBasic::InitExecutor(const std::string &device_name, uint32_t device_id) {
  device_id_ = device_id;
  context_ = std::make_shared<Context>(device_name, device_id);
  executor_ = ExecutorManager::Instance().GetExecutor(device_name, device_id);
}

GraphId SessionBasic::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
  for (const auto &graph_item : graphs_) {
    auto graph = graph_item.second;
    MS_EXCEPTION_IF_NULL(graph);
    // if front_anf is a parameter,the backend parameter may have two
    if (graph->GetBackendAnfByFrontAnf(front_anf) != nullptr) {
      return graph_item.first;
    }
  }
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_LOG(DEBUG) << "Front_anf " << front_anf->DebugString() << " is not exist in any graph";
  return kInvalidGraphId;
}

KernelGraphPtr SessionBasic::GetGraph(mindspore::GraphId graph_id) const {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(INFO) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void SessionBasic::ClearGraph() {
  auto graph_iter = graphs_.begin();
  while (graph_iter != graphs_.end()) {
    graph_iter->second.reset();
    graphs_.erase(graph_iter++);
  }
  graph_sum_ = 0;
}

void SessionBasic::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) {
  auto graph_id = GetGraphIdByNode(out_node);
  if (graph_id == kInvalidGraphId) {
    return;
  }
  auto node_graph = GetGraph(graph_id);
  if (node_graph == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Init parameter with pre graph output node: " << out_node->DebugString();
  auto ref_node = node_graph->GetInternalOutputByFrontNode(out_node);
  if (ref_node == nullptr) {
    MS_LOG(INFO) << "No corresponding internal output for output node";
    return;
  }
  size_t output_idx = 0;
  if (AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  auto ref_real_node_index = real_kernel.second;
  if (ref_real_node->isa<CNode>() && node_graph->IsUniqueTargetInternalOutput(ref_real_node, ref_real_node_index)) {
    auto kernel_info = ref_real_node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      MS_LOG(INFO) << "No kernel info";
      return;
    }
    if (!opt::IsNopNode(ref_real_node) && !AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index)) {
      MS_LOG(INFO) << "No kernel address";
      return;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    auto d_kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(d_kernel_info);
    parameter->set_kernel_info(d_kernel_info);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    d_kernel_info->set_select_kernel_build_info(builder.Build());
    AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type),
                                                               parameter->Shape()->cast<abstract::BaseShapePtr>());
    parameter->set_abstract(abstract);
  }
}

AnfNodePtr SessionBasic::CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (IgnoreCreateParameterForMakeTuple(node)) {
    return nullptr;
  }
  auto new_parameter = graph->TransTupleToMakeTuple(graph->NewParameter(node->abstract()));
  auto parameters = AnfAlgo::GetAllOutput(new_parameter);
  std::vector<AnfNodePtr> pre_graph_out = {node};
  // If a cnode is a call, it's input0 is a cnode too, so it doesn't have primitive
  if (!pre_graph_out.empty() && !AnfAlgo::IsRealKernel(node)) {
    pre_graph_out = AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem, prim::kPrimUpdateState});
  }
  for (const auto &parameter : parameters) {
    auto valid_inputs = graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    valid_inputs->push_back(true);
    graph_inputs->push_back(parameter);
  }
  size_t param_index = 0;
  for (const auto &out_node : pre_graph_out) {
    size_t output_size = AnfAlgo::GetOutputTensorNum(out_node);
    for (size_t i = 0; i < output_size; i++) {
      if (param_index >= parameters.size()) {
        MS_LOG(EXCEPTION) << "Parameters size:" << parameters.size() << "out of range.Node:" << node->DebugString()
                          << ",out_node:" << out_node->DebugString();
      }
      InitInternalOutputParameter(out_node, parameters[param_index++]);
    }
  }
  return new_parameter;
}

ParameterPtr SessionBasic::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_value = GetParamDefaultValue(anf);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ParamInfoPtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
  }
  new_parameter->IncreaseUsedGraphCount();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(true);
  return new_parameter;
}

AnfNodePtr SessionBasic::CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  return CreateParameterFromTuple(anf, graph);
}

void SessionBasic::GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    // push attr to inputs[0] of new cnode
    cnode_inputs->push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
  } else {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs->push_back(std::make_shared<ValueNode>(new_fg));
  }
}

void SessionBasic::GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                                     std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto origin_inputs = cnode->inputs();
  bool optimize_depend = IsPrimitiveCNode(cnode, prim::kPrimDepend) && origin_inputs.size() >= 3;
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < origin_inputs.size(); input_idx++) {
    auto anf = origin_inputs[input_idx];
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
      continue;
    } else if (optimize_depend && input_idx > 1) {
      cnode_inputs->push_back(NewValueNode(MakeValue(SizeToInt(input_idx))));
      continue;
    } else if (other_graph_cnode->find(anf) != other_graph_cnode->end()) {
      cnode_inputs->push_back((*other_graph_cnode)[anf]);
      continue;
    } else if (anf->isa<ValueNode>() && !IsValueNode<FuncGraph>(anf)) {
      // if input is a value node,
      auto new_value_node = CreateNewValueNode(anf, graph);
      if (new_value_node != nullptr) {
        cnode_inputs->emplace_back(new_value_node);
      }
      continue;
    } else if (anf->isa<Parameter>()) {
      auto new_parameter = CreateNewParameterFromParameter(anf, graph);
      cnode_inputs->push_back(new_parameter);
      if (GetGraphIdByNode(anf) == kInvalidGraphId) {
        graph->FrontBackendlMapAdd(anf, new_parameter);
      } else {
        (*other_graph_cnode)[anf] = new_parameter;
      }
      continue;
    } else {
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, graph);
      if (parameter_from_cnode == nullptr) {
        parameter_from_cnode = NewValueNode(MakeValue(SizeToLong(input_idx)));
      }
      if (parameter_from_cnode->isa<Parameter>() && IsPrimitiveCNode(anf, prim::kPrimLoad)) {
        auto para = parameter_from_cnode->cast<ParameterPtr>();
        auto load_cnode = anf->cast<CNodePtr>();
        para->set_name(load_cnode->input(kFirstDataInputIndex)->fullname_with_scope());
      }
      cnode_inputs->push_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
    }
  }
}

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                                      std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  // get primitive of old node
  std::vector<AnfNodePtr> cnode_inputs;
  GetCNodeInfo(cnode, &cnode_inputs);
  GetNewCNodeInputs(cnode, graph, &cnode_inputs, other_graph_cnode);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  return new_cnode;
}

CNodePtr SessionBasic::CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto partial_node = graph->GetBackendAnfByFrontAnf(node_input);
    return partial_node->cast<CNodePtr>();
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

std::vector<AnfNodePtr> SessionBasic::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  if (cnode->inputs().size() < 2) {
    cnode_inputs = switch_cnode->inputs();
    return cnode_inputs;
  }
  std::vector<AnfNodePtr> switch_inputs = {switch_cnode->input(kAnfPrimitiveIndex),
                                           switch_cnode->input(kFirstDataInputIndex)};
  for (size_t index = kFirstBranchInSwitch; index < switch_cnode->inputs().size(); index++) {
    auto node = switch_cnode->input(index);
    // there is real input in call, should put it to true and false branch in switch
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      auto partial_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      std::vector<AnfNodePtr> partial_inputs = partial_node->inputs();
      // Put all call args at the end of partial inputs.
      for (size_t i = kFirstDataInputIndex; i < cnode->size(); ++i) {
        partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(i)));
      }
      auto new_partial = graph->NewCNode(partial_inputs);
      switch_inputs.emplace_back(new_partial);
    }
  }
  if (switch_inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch inputs size: " << switch_inputs.size() << "less than " << kSwitchInputSize;
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  cnode_inputs.emplace_back(switch_node);
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
  if (AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial)) {
    auto return_input_cnode = return_input->cast<CNodePtr>();
    auto partial_inputs = return_input_cnode->inputs();
    call_inputs.insert(call_inputs.end(), partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end());
  } else if (IsValueNode<KernelGraph>(return_input)) {  // return node is kernel graph
    call_inputs.emplace_back(return_input);
  } else {  // return node is value node
    KernelGraphPtr kernel_graph = NewKernelGraph();
    auto valid_inputs = kernel_graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = kernel_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    std::vector<AnfNodePtr> cnode_inputs = {return_input};
    for (auto real_input : real_inputs) {
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
  for (auto real_input : real_inputs) {
    auto parameter_for_input = CreateNewParameterFromCNode(real_input, graph);
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
  auto make_tuple_node = switch_layer_cnode->input(kMakeTupleInSwitchLayerIndex);
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
    if (AnfAlgo::CheckPrimitiveType(partial_idx, prim::kPrimPartial)) {
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
    if (AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial) || return_input->isa<ValueNode>()) {
      ProcessNodeRetFunc(cnode, partial_kernel_graph.get(), real_inputs);
    }
    // partial node add input args
    new_partial_inputs.insert(new_partial_inputs.end(), real_inputs.begin(), real_inputs.end());
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
  if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                   std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                     MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                     return graph->GetBackendAnfByFrontAnf(node);
                   });
    return cnode_inputs;
  } else if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    return CreateCallSwitchInputs(cnode, graph);
  } else if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitchLayer)) {
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
  if (AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
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
  if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
    for (size_t index = kFirstBranchInSwitch; index < cnode->inputs().size(); index++) {
      auto node_input = cnode->input(index);
      auto switch_input = CreateSwitchInput(cnode, node_input, graph);
      cnode_inputs->emplace_back(switch_input);
    }
  } else {
    for (size_t input_idx = kFirstDataInputIndex; input_idx < cnode->inputs().size(); input_idx++) {
      auto anf = cnode->input(input_idx);
      MS_EXCEPTION_IF_NULL(anf);
      // anf has been created before
      if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
        cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
        continue;
      } else if (IsValueNode<None>(anf)) {
        continue;
      }
      MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
    }
  }
}

CNodePtr SessionBasic::CreateNewCNode(CNodePtr cnode, KernelGraph *graph) {
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
    auto prim = AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }
  // handle inputs of cnode except primitive
  CreateCNodeInputs(cnode, graph, &cnode_inputs);

  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);

  // if the cnode is call switch, remove call
  if (new_cnode->inputs().size() > 1) {
    auto first_input = new_cnode->input(kFirstDataInputIndex);
    MS_EXCEPTION_IF_NULL(first_input);
    if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitch)) {
      new_cnode = first_input->cast<CNodePtr>();
    }
    if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitchLayer)) {
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
  auto sub_func_graph = AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendlMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr SessionBasic::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ParamInfoPtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
  }
  new_parameter->IncreaseUsedGraphCount();

  return new_parameter;
}

KernelGraphPtr SessionBasic::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  std::unordered_map<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  for (const auto &node : lst) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // create a new cnode object
    auto new_cnode = CreateNewCNode(cnode, graph.get(), &other_graph_cnode);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      new_cnode->set_fullname_with_scope(cnode->input(kFirstDataInputIndex)->fullname_with_scope());
    }
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendlMapAdd(node, new_cnode);
  }
  // add a make_tuple at the end of graph as output
  graph->set_output(ConstructOutput(outputs, graph));
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }

  // Update Graph Dynamic Shape Attr
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));
  UnifyMindIR(graph);
  opt::BackendCommonOptimization(graph);
  graph->SetInputNodes();
  auto input_nodes = graph->input_nodes();
  for (auto input_node : input_nodes) {
    if (input_node->isa<Parameter>()) {
      auto node_ptr = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(node_ptr);
      if (!IsUsedByRealKernel(manager, input_node)) {
        node_ptr->set_used_by_real_kernel(false);
      }
      auto shape = node_ptr->Shape();
      if (IsShapeDynamic(shape->cast<abstract::ShapePtr>())) {
        node_ptr->set_used_by_dynamic_kernel(true);
      }
    }
  }
  graph->SetOptimizerFlag();
  return graph;
}

GraphInfo SessionBasic::GetSingleOpGraphInfo(const CNodePtr &kernel,
                                             const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto prim = AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  const AbstractBasePtr &abstract = kernel->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
  GraphInfo graph_info;
  // get input tensor info
  for (const auto &tensor : input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_shape = tensor->shape();
    (void)std::for_each(tensor_shape.begin(), tensor_shape.end(),
                        [&](const auto &dim) { (void)graph_info.append(std::to_string(dim) + "_"); });
    (void)graph_info.append(std::to_string(tensor->data_type()) + "_");
    if (tensor->device_address() != nullptr) {
      const auto type_id = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->type_id();
      (void)graph_info.append(std::to_string(type_id) + "_");
      const auto format = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->format();
      (void)graph_info.append(format + "_");
    }
    for (const auto &padding_type : tensor->padding_type()) {
      (void)graph_info.append(std::to_string(padding_type) + "_");
    }
  }
  // get attr info
  const auto &attr_map = prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(), [&](const auto &element) {
    if (element.second->ToString().empty()) {
      return;
    }
    (void)graph_info.append(element.second->ToString() + "_");
  });
  auto build_shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(build_shape);
  (void)graph_info.append(build_shape->ToString() + "_");
  for (size_t output_index = 0; output_index < output_num; output_index += 1) {
    const auto output_type = AnfAlgo::GetOutputInferDataType(kernel, output_index);
    (void)graph_info.append(std::to_string(output_type) + "_");
  }
  graph_info.append(prim->id());
  return graph_info;
}

void SessionBasic::GetSingleOpRunInfo(const CNodePtr cnode, OpRunInfo *run_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(run_info);
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  run_info->primitive = primitive;
  run_info->op_name = primitive->name();
  const auto &abstract = cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract is nullptr, node = " << cnode->DebugString();
  }
  run_info->abstract = abstract;
  const auto &shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  const auto &shape_info = shape->ToString();
  if (shape_info.find("-1") != string::npos) {
    run_info->is_dynamic_shape = true;
  }
}

TensorPtr SessionBasic::GetValueNodeOutputTensor(const AnfNodePtr &node, size_t output_index) {
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
  }
  return nullptr;
}

TensorPtr SessionBasic::GetParameterOutputTensor(const AnfNodePtr &node,
                                                 const std::map<AnfNodePtr, size_t> &parameter_index,
                                                 const std::vector<tensor::TensorPtr> &graph_inputs) {
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
                                             const std::map<KernelWithIndex, tensor::TensorPtr> &op_output) {
  const auto &iter = op_output.find(kernel_with_index);
  if (iter == op_output.end()) {
    MS_LOG(EXCEPTION) << "Can not find output tensor of cnode, node = " << kernel_with_index.first->DebugString();
  }
  return iter->second;
}

void SessionBasic::GetOpInputTensors(const CNodePtr &cnode,
                                     const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                                     const std::map<AnfNodePtr, size_t> &parameter_index,
                                     const std::vector<tensor::TensorPtr> &graph_inputs,
                                     InputTensorInfo *input_tensor_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_tensor_info);
  for (size_t i = 1; i < cnode->inputs().size(); i += 1) {
    const auto &input = cnode->input(i);
    auto kernel_with_index = AnfAlgo::VisitKernel(input, 0);
    auto real_input = kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    tensor::TensorPtr tensor = nullptr;
    if (real_input->isa<ValueNode>()) {
      if (HasAbstractMonad(real_input)) {
        continue;
      }
      tensor = GetValueNodeOutputTensor(real_input, kernel_with_index.second);
    } else if (real_input->isa<Parameter>()) {
      tensor = GetParameterOutputTensor(real_input, parameter_index, graph_inputs);
    } else if (real_input->isa<CNode>()) {
      tensor = GetCNodeOutputTensor(kernel_with_index, op_output);
      input_tensor_info->input_kernel.insert(kernel_with_index);
    } else {
      MS_LOG(EXCEPTION) << "Invalid input node, node = " << real_input->DebugString();
    }
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Get" << i << "th input tensor of " << cnode->fullname_with_scope() << " from "
                  << real_input->fullname_with_scope() << "-" << kernel_with_index.second;
    input_tensor_info->input_tensors_mask.emplace_back(tensor->is_parameter() ? kParameterWeightTensorMask
                                                                              : kParameterDataTensorMask);
    input_tensor_info->input_tensors.emplace_back(tensor);
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
  graph->FrontBackendlMapAdd(node, new_cnode);
  if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimReturn)) {
    graph->set_return(new_cnode);
  }
  return true;
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                std::vector<KernelGraphPtr> *all_out_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph] = graph;
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    // Create parameter
    if (node->isa<Parameter>()) {
      auto graph_inputs = graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(graph_inputs);
      auto new_parameter = CreateNewParameter(node, graph.get());
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendlMapAdd(node, new_parameter);
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
      FuncGraphPtr child_graph = AnfAlgo::GetValueNodeFuncGraph(node);
      if (front_backend_graph_map_.find(child_graph) == front_backend_graph_map_.end()) {
        (void)ConstructKernelGraph(child_graph, all_out_graph);
      }
      (void)CreateValueNodeKernelGraph(node, graph.get());
      auto &parent_graph = parent_graphs_[front_backend_graph_map_[child_graph]->graph_id()];
      auto parent_graph_it =
        std::find(parent_graph.begin(), parent_graph.end(), front_backend_graph_map_[func_graph]->graph_id());
      if (parent_graph_it == parent_graph.end()) {
        parent_graph.push_back(front_backend_graph_map_[func_graph]->graph_id());
      }
      continue;
    }
    // Create cnode
    if (!CreateCNodeOfKernelGraph(node, graph.get())) {
      DumpIR("construct_kernel_graph_fail.ir", func_graph);
      MS_LOG(EXCEPTION) << "Construct func graph " << func_graph->ToString() << " failed."
                        << trace::DumpSourceLines(node);
    }
  }

  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  FuncGraphManagerPtr manager = MakeManager({graph});
  auto input_nodes = graph->inputs();
  for (auto input_node : input_nodes) {
    if (input_node->isa<Parameter>()) {
      auto node_ptr = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(node_ptr);
      if (!IsUsedByRealKernel(manager, input_node)) {
        node_ptr->set_used_by_real_kernel(false);
      }
      auto shape = node_ptr->Shape();
      if (IsShapeDynamic(shape->cast<abstract::ShapePtr>())) {
        node_ptr->set_used_by_dynamic_kernel(true);
      }
    }
  }
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
  all_out_graph->push_back(graph);
  return graph;
}

void SessionBasic::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) {
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
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    graph_inputs->push_back(backend_parameter);
  }
}

namespace {
bool TensorNeedSync(const AnfNodePtr &parameter, const tensor::TensorPtr &tensor) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_address = AnfAlgo::GetMutableOutputAddr(parameter, 0);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    return tensor->device_address().get() == nullptr || tensor->device_address() != device_address;
  }
  if (tensor->NeedSyncHostToDevice()) {
    return true;
  }
  auto tensor_address = tensor->device_address();
  if (tensor_address != device_address) {
    tensor->data_sync(false);
    return true;
  }
  return false;
}
}  // namespace

// run graph steps
void SessionBasic::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs_const) const {
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 3;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->input_ctrl_tensors()) {
    input_ctrl_size = LoadCtrlInputTensor(kernel_graph, &inputs);
  }
  auto &input_nodes = kernel_graph->input_nodes();
  auto extra_param_size = kernel_graph->GetExtraParamAndTensor().size();
  if ((inputs.size() + input_ctrl_size) - 3 != input_nodes.size() - extra_param_size) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size()
                      << ", input_ctrl_size:" << input_ctrl_size << ", extra_param_size:" << extra_param_size;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    auto size = LongToSize(tensor->data().nbytes());
    if (input_node->isa<Parameter>() && input_node->cast<ParameterPtr>()->is_used_by_dynamic_kernel()) {
      auto tensor_shape = tensor->shape();
      std::vector<size_t> shape_tmp;
      (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(shape_tmp), IntToSize);
      AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input_node, 0)}, {shape_tmp},
                                          input_node.get());
      size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(tensor->data_type());
    }
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0) && TensorNeedSync(input_node, tensor)) {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
      const std::string &param_name = input_node->fullname_with_scope();
      if (ps::ps_cache_instance.IsHashTable(param_name)) {
        continue;
      }
#endif
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      if (size != 0 && !device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(input_node, 0), size,
                                                         tensor->data_type(), tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }

      if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode ||
          AnfAlgo::IsParameterWeight(input_node->cast<ParameterPtr>())) {
        tensor->set_device_address(device_address);
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
}

void SessionBasic::UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                                 const std::vector<tensor::TensorPtr> &input_tensors) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Update output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, &tensor_to_node));
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (auto &item : tensor_to_node) {
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
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
      tensor->data_sync(false);
      tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void SessionBasic::UpdateOutputAbstract(const std::shared_ptr<KernelGraph> &kernel_graph,
                                        OpRunInfo *op_run_info) const {
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

std::vector<tensor::TensorPtr> SessionBasic::GetInputNeedLockTensors(const GraphId &graph_id,
                                                                     const std::vector<tensor::TensorPtr> &inputs) {
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_optimizer()) {
    return {};
  }
  std::vector<tensor::TensorPtr> result;
  for (auto &tensor : inputs) {
    if (!tensor->IsGraphOutput()) {
      result.emplace_back(tensor);
    }
  }
  return result;
}

void SessionBasic::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                       VectorRef *outputs,
                                       std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, tensor_to_node));
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
  vector<ParameterPtr> paras;
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    if (!kernel_graph_inputs[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      continue;
    }
    auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
    if (!AnfAlgo::IsParameterWeight(parameter)) {
      vector<int64_t> input_shape;
      auto parameter_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
      (void)std::transform(parameter_shape.begin(), parameter_shape.end(), std::back_inserter(input_shape),
                           [](const size_t dim) { return SizeToLong(dim); });
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
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    vector_outputs.emplace_back(CreateNodeOutputTensors(item, kernel_graph, inputs, &tensor_to_node));
  }
  *outputs = TransformVectorRefToMultiTensor(vector_outputs);
  for (size_t i = 0; i < outputs->size(); i++) {
    output_names->push_back("output" + std::to_string(i));
  }
}

void SessionBasic::RegisterSummaryCallBackFunc(const CallBackFunc &callback) {
  MS_EXCEPTION_IF_NULL(callback);
  summary_callback_ = callback;
}

void SessionBasic::RunInfer(NotNull<FuncGraphPtr> func_graph, const std::vector<tensor::TensorPtr> &inputs) {
  auto node_list = TopoSort(func_graph->get_return());
  size_t tensor_index = 0;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      AbstractBasePtrList input_abstracts;
      size_t input_num = AnfAlgo::GetInputTensorNum(node);
      for (size_t index = 0; index < input_num; ++index) {
        auto input_node = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), index);
        MS_EXCEPTION_IF_NULL(input_node);
        auto abstract = input_node->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        input_abstracts.emplace_back(abstract);
      }
      auto prim = AnfAlgo::GetCNodePrimitive(node);
      if (prim->isa<ops::PrimitiveC>()) {
        auto prim_c = prim->cast<std::shared_ptr<ops::PrimitiveC>>();
        MS_EXCEPTION_IF_NULL(prim_c);
        auto abstract = prim_c->Infer(input_abstracts);
        node->set_abstract(abstract);
      }
    } else if (node->isa<Parameter>()) {
      if (tensor_index > inputs.size()) {
        MS_EXCEPTION(IndexError) << "Index " << tensor_index << "is out of " << inputs.size() << "tensor's size";
      }
      node->set_abstract(inputs[tensor_index++]->ToAbstract());
    } else {
      auto value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      value_node->set_abstract(value->ToAbstract());
    }
  }
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
        MS_LOG(EXCEPTION) << "The node Summary should have 2 inputs at least!";
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (!AnfAlgo::IsRealKernel(item_with_index.first)) {
        MS_LOG(EXCEPTION) << "Unexpected node:" << item_with_index.first->DebugString();
      }
      summary[n->fullname_with_scope()] = item_with_index;
    }
  }
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
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
    MS_LOG(ERROR) << "The Summary operator can not collect data correctly. Detail: the data sink mode is used and the"
                     " sink size(in model.train() python api) is not equal to 1.";
  }
  SetSummaryNodes(graph);
  auto summary_outputs = graph->summary_nodes();
  std::map<std::string, tensor::TensorPtr> params_list;
  // fetch outputs apply kernel in session & run callback functions
  for (auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetOutputAddr(node, index);
    auto shape = AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = AnfAlgo::GetOutputInferDataType(node, index);
    std::vector<int64_t> temp_shape;
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
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

namespace {
bool CNodeFirstInputIsPrimitive(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto prim = cnode->input(kAnfPrimitiveIndex);
  if (prim == nullptr || !IsValueNode<Primitive>(prim)) {
    return false;
  }
  return true;
}

std::vector<AnfNodePtr> ExtendNodeUsers(const FuncGraphManagerPtr &front_func_graph_manager,
                                        const AnfNodePtr &front_node) {
  auto &users = front_func_graph_manager->node_users()[front_node];
  std::vector<AnfNodePtr> result;
  for (auto &user : users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimControlDepend)) {
      continue;
    }
    if (IsPrimitiveCNode(user.first, prim::kPrimDepend)) {
      auto depend_cnode = user.first->cast<CNodePtr>();
      if (depend_cnode == nullptr) {
        continue;
      }
      if (front_node != depend_cnode->input(1)) {
        continue;
      }
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      result.insert(result.end(), res.begin(), res.end());
      continue;
    }
    result.emplace_back(user.first);
  }
  return result;
}

AnfNodePtr GetSupportedInternalNode(const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    return nullptr;
  }
  if (AnfAlgo::IsRealKernel(front_node)) {
    return front_node;
  }
  if (AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimTupleGetItem)) {
    return front_node;
  }
  if (AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimDepend)) {
    auto cnode = front_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.size() > 2) {
      return GetSupportedInternalNode(inputs[1]);
    }
  }
  return nullptr;
}

void HandleInternalOutput(const AnfNodePtr &input_front_node, const AnfNodePtr &backend_node,
                          const FuncGraphManagerPtr &front_func_graph_manager,
                          const std::shared_ptr<KernelGraph> &backend_graph) {
  auto front_node = GetSupportedInternalNode(input_front_node);
  if (front_node == nullptr) {
    return;
  }
  auto front_real_kernel_pair = AnfAlgo::VisitKernel(front_node, 0);
  auto backend_real_kernel_pair = AnfAlgo::VisitKernel(backend_node, 0);
  auto backend_real_kernel = backend_real_kernel_pair.first;
  if (backend_real_kernel == nullptr || !backend_real_kernel->isa<CNode>()) {
    return;
  }
  auto front_real_kernel = front_real_kernel_pair.first;
  std::string kernel_target = GetCNodeTarget(front_real_kernel);
  bool internal_output = CNodeFirstInputIsPrimitive(front_real_kernel);
  bool unique_target = true;
  if (internal_output && opt::IsNopNode(front_real_kernel)) {
    auto pre_node_pair = AnfAlgo::GetPrevNodeOutput(front_real_kernel, 0);
    auto pre_node_target = GetCNodeTarget(pre_node_pair.first);
    if (pre_node_target != kernel_target) {
      unique_target = false;
    }
  }
  if (internal_output) {
    auto users = ExtendNodeUsers(front_func_graph_manager, front_node);
    for (auto user : users) {
      if (!CNodeFirstInputIsPrimitive(user)) {
        internal_output = false;
        break;
      }
      if (!AnfAlgo::IsRealKernel(user)) {
        internal_output = false;
        break;
      }
      if (kernel_target != GetCNodeTarget(user)) {
        unique_target = false;
      }
    }
  }
  if (internal_output) {
    MS_LOG(INFO) << "AddInternalOutput: " << front_node->DebugString() << " To " << backend_real_kernel->DebugString()
                 << ", unique_target: " << unique_target;
    backend_graph->AddInternalOutput(front_node, backend_real_kernel, backend_real_kernel_pair.second, unique_target);
  }
}
}  // namespace

CNodePtr SessionBasic::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "Output:" << output->DebugString();
  }
  auto FindEqu = [graph, outputs](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
        return backend_anf;
      }

      MS_EXCEPTION_IF_NULL(out);
      auto out_func_graph = out->func_graph();
      MS_EXCEPTION_IF_NULL(out_func_graph);
      auto out_func_graph_manager = out_func_graph->manager();
      if (out_func_graph_manager == nullptr) {
        return backend_anf;
      }
      HandleInternalOutput(out, backend_anf, out_func_graph_manager, graph);
      return backend_anf;
    }
    MS_LOG(EXCEPTION) << "Can't find the node in the equiv map!";
  };
  output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_args),
                       [&](const AnfNodePtr &out) -> AnfNodePtr { return FindEqu(out); });
  return graph->NewCNode(output_args);
}

void SessionBasic::CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) {
  MS_LOG(INFO) << "Start!";
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfRuntimeAlgorithm::GetOutputTensorNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfRuntimeAlgorithm::GetOutputTensorNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToLong(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int64Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
      std::vector<TypeId> types = {AnfAlgo::GetOutputInferDataType(cnode, output_index)};
      std::vector<std::vector<size_t>> shapes = {AnfAlgo::GetOutputInferShape(cnode, output_index)};
      AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
      make_tuple_inputs.push_back(getitem);
    }
  } else {
    make_tuple_inputs.push_back(cnode);
  }
  // create output
  auto g_output = graph->NewCNode(make_tuple_inputs);
  graph->set_output(g_output);
  MS_LOG(INFO) << "Finish!";
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const OpRunInfo &op_run_info,
                                                                  const std::vector<tensor::TensorPtr> &input_tensors,
                                                                  const std::vector<int64_t> &tensors_mask,
                                                                  bool is_ascend) {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graph_sum_++;
  std::vector<AnfNodePtr> inputs;
  // set input[0]
  PrimitivePtr op_prim = op_run_info.primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  inputs.push_back(std::make_shared<ValueNode>(op_prim));
  // set input parameter
  MS_LOG(INFO) << "Input tensor size: " << input_tensors.size();
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (tensors_mask[i] == kValueNodeTensorMask) {
      auto value_node = ConstructRunOpValueNode(graph, input_tensors[i]);
      inputs.push_back(value_node);
      continue;
    }
    auto parameter = ConstructRunOpParameter(graph, input_tensors[i], tensors_mask[i]);
    inputs.push_back(parameter);
    auto mutable_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(mutable_inputs);
    mutable_inputs->push_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info.abstract);
  // get output dynamic shape info
  AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(op_run_info.is_dynamic_shape), cnode);
  if (op_run_info.is_auto_mixed_precision) {
    AnfAlgo::SetNodeAttr(kAttrPynativeNextOpName, MakeValue(op_run_info.next_op_name), cnode);
    AnfAlgo::SetNodeAttr(kAttrPynativeNextIndex, MakeValue(op_run_info.next_input_index), cnode);
  }
  // set execution order
  std::vector<CNodePtr> exe_order = {cnode};
  graph->set_execution_order(exe_order);
  graph->UpdateGraphDynamicAttr();
  // set output
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
  return graph;
}

KernelGraphPtr SessionBasic::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = graph;
  return graph;
}

AnfNodePtr SessionBasic::FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list) {
  MS_EXCEPTION_IF_NULL(push_node);
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      for (auto input : node->cast<CNodePtr>()->inputs()) {
        if (push_node == AnfAlgo::VisitKernel(input, 0).first) {
          if (AnfAlgo::GetCNodeName(node) != kPullOpName) {
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

void SessionBasic::RunOp(OpRunInfo *op_run_info, const GraphInfo &graph_info,
                         std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                         const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunOp(shared_from_this(), op_run_info, graph_info, input_tensors, outputs, tensors_mask);
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

void SessionBasic::RunOpsInGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                     VectorRef *outputs) {
  MS_LOG(INFO) << "Start!";
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::map<AnfNodePtr, size_t> parameter_index;
  GetParameterIndex(kernel_graph.get(), inputs, &parameter_index);
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> output_indexes;
  CreateOutputPlaceholder(kernel_graph, inputs, outputs, &output_indexes);
  std::map<KernelWithIndex, size_t> cnode_refcount;
  GetRefCount(kernel_graph.get(), &cnode_refcount);
  BuildOpsInGraph(graph_id, parameter_index, inputs, cnode_refcount);

  // Clear bucket resources every step
  if (kernel_graph->is_bprop()) {
    ClearAllBucket(graph_id);
  }

  std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;
  for (const auto &kernel : kernel_graph->execution_order()) {
    // Generate input tensors, tensor masks and input kernel with index
    InputTensorInfo input_tensor_info;
    GetOpInputTensors(kernel, op_output_map, parameter_index, inputs, &input_tensor_info);

    // Get OpRunInfo and GraphInfo
    OpRunInfo run_info;
    GetSingleOpRunInfo(kernel, &run_info);
    GraphInfo graph_info = GetSingleOpGraphInfo(kernel, input_tensor_info.input_tensors);

    // Build and run current single op
    VectorRef op_outputs;
    RunOpImpl(graph_info, &run_info, &input_tensor_info.input_tensors, &op_outputs,
              input_tensor_info.input_tensors_mask);

    std::vector<tensor::TensorPtr> new_output_tensors;

    // Handle inputs and outputs of current op
    HandleOpInputs(input_tensor_info.input_kernel, &cnode_refcount, &op_output_map);
    HandleOpOutputs(kernel, op_outputs, output_indexes, cnode_refcount, &op_output_map, outputs, &new_output_tensors);
    // Save grad node to Bucket
    if (kernel_graph->is_bprop()) {
      AddGradAddrToBucket(graph_id, new_output_tensors);
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void SessionBasic::EraseValueNodeTensor(const std::vector<int64_t> &tensors_mask,
                                        std::vector<tensor::TensorPtr> *input_tensors) {
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

void SessionBasic::UpdateAllGraphDynamicShapeAttr(const std::vector<KernelGraphPtr> &all_graphs) {
  bool is_dynamic = false;
  for (const auto &graph : all_graphs) {
    UpdateGraphDynamicShapeAttr(NOT_NULL(graph));
    is_dynamic = graph->is_dynamic_shape() || is_dynamic;
  }
  if (is_dynamic && all_graphs.size() > 1) {
    MS_LOG(EXCEPTION) << "Dynamic shape is not supported with control flow.";
  }
}

void SessionBasic::UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &root_graph) {
  for (const auto &cnode : root_graph->execution_order()) {
    if (AnfAlgo::IsNodeDynamicShape(cnode)) {
      AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
  }
  root_graph->UpdateGraphDynamicAttr();
}

bool SessionBasic::IsGetNextGraph(const GraphId &graph_id, std::string *channel_name) {
  auto kernel_graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kGetNextOpName) {
      auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
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

void SessionBasic::RunOpHideNopNode(const KernelGraphPtr &kernel_graph) const {
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
  return AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem}).size();
}

void SetGraphBpropAttr(const KernelGraphPtr &graph) {
  auto &execution_orders = graph->execution_order();
  if (std::any_of(execution_orders.begin(), execution_orders.end(),
                  [](const AnfNodePtr &node) { return node->scope()->name().rfind("Gradient", 0) == 0; })) {
    graph->set_is_bprop(true);
    MS_LOG(INFO) << "Match bprop graph";
  } else {
    graph->set_is_bprop(false);
  }
}

std::vector<uint32_t> GenerateBucketSizeList(const KernelGraphPtr &graph, const std::vector<uint32_t> &split_index) {
  if (split_index.empty()) {
    auto grads_count = GetBpropGraphGradsCount(graph);
    if (grads_count == 0) {
      MS_LOG(EXCEPTION) << "Bprop graph has no grad";
    }
    return {grads_count};
  }

  std::vector<uint32_t> bucket_size_list;
  uint32_t old_index = 0;
  for (auto &index : split_index) {
    if (old_index == 0) {
      bucket_size_list.emplace_back(index - old_index + 1);
    } else {
      bucket_size_list.emplace_back(index - old_index);
    }
    old_index = index;
  }
  return bucket_size_list;
}

void PreProcessOnSplitIndex(const KernelGraphPtr &graph, vector<uint32_t> *split_index) {
  MS_EXCEPTION_IF_NULL(split_index);
  if (split_index->empty()) {
    return;
  }
  // calculate split index num
  auto split_index_len = split_index->size();
  uint32_t split_index_num = (*split_index)[split_index_len - 1];
  // obtain graph output tensor num
  auto grads_count = GetBpropGraphGradsCount(graph);
  if (split_index_num == 0 || split_index_num >= grads_count) {
    MS_LOG(EXCEPTION) << "invalid allreduce split index " << split_index_num << " and grads count " << grads_count;
  }
  if (split_index_num < grads_count - 1) {
    split_index->push_back(grads_count - 1);
  }
}

void SessionBasic::InitAllBucket(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Init Bucket start, graph_id:" << graph->graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (!pynative_mode || parallel_mode != parallel::DATA_PARALLEL) {
    return;
  }
  SetGraphBpropAttr(graph);

  if (!graph->is_bprop()) {
    return;
  }

  std::vector<std::shared_ptr<device::Bucket>> bucket_list;
  // Create bucket for every split allreduce ops
  auto split_index = GetAllReduceSplitIndex();
  PreProcessOnSplitIndex(graph, &split_index);
  auto bucket_size_list = GenerateBucketSizeList(graph, split_index);
  uint32_t bucket_id = 0;
  for (auto bucket_size : bucket_size_list) {
    MS_LOG(INFO) << "Create new bucket:" << bucket_id;
    auto bucket = CreateBucket(bucket_id++, bucket_size);
    bucket->Init();
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
  MS_LOG(INFO) << "Init Bucket finish";
}

void SessionBasic::AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::DATA_PARALLEL) {
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

void SessionBasic::DumpGraph(const std::shared_ptr<KernelGraph> &kernel_graph) {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIR("graph_build_" + std::to_string(kernel_graph->graph_id()) + ".ir", kernel_graph, true, kWholeStack);
    DumpIRProto(kernel_graph, "vm_build_" + std::to_string(kernel_graph->graph_id()));
    DumpIR("trace_code_graph", kernel_graph, true, kWholeStack);
  }
#endif
}

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
void SessionBasic::InitPsWorker(const KernelGraphPtr &kernel_graph) {
  if (!ps::PSContext::instance()->is_worker()) {
    return;
  }
  CheckPSModeConsistence(kernel_graph);
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    if (!ps::ps_cache_instance.initialized_ps_cache()) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      auto devcie_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(devcie_target, device_id_);
      MS_EXCEPTION_IF_NULL(runtime_instance);
      auto context = runtime_instance->context();
      const auto &kernels = kernel_graph->execution_order();
      if (kernels.size() > 0 && AnfAlgo::GetCNodeName(kernels[0]) == "InitDataSetQueue") {
        GetBatchElements(kernels[0]);
        ps::ps_cache_instance.Initialize();
      }
      ps::ps_cache_instance.DoProcessData(device_id_, context);
    }
  } else {
    // Assign parameter keys.
    AssignParamKey(kernel_graph);
  }
}

void SessionBasic::GetBatchElements(const AnfNodePtr &kernel_node) const {
  auto shapes = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");
  auto types = AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "types");
  if (shapes.size() != types.size() || shapes.size() == 0 || types.size() == 0) {
    MS_LOG(EXCEPTION) << "Invalid shapes of op[InitDataSetQueue]: shapes size " << shapes.size() << ", types size "
                      << types;
  }
  size_t batch_elements = 1;
  const auto &shape = shapes[0];
  for (size_t i = 0; i < shape.size(); ++i) {
    batch_elements *= shape[i];
  }
  ps::ps_cache_instance.set_batch_elements(batch_elements);
}

void SessionBasic::CheckPSModeConsistence(const KernelGraphPtr &kernel_graph) const {
  auto input_nodes = kernel_graph->inputs();
  for (const auto &input_node : input_nodes) {
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    auto pk_node = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto param_info_ptr = pk_node->param_info();
    const std::string &param_name = pk_node->fullname_with_scope();
    if (param_info_ptr != nullptr && param_info_ptr->init_in_server() &&
        !ps::ps_cache_instance.IsHashTable(param_name)) {
      MS_LOG(EXCEPTION) << "Can not initialize the parameter[" << param_name
                        << "] in server, this parameter is used by kernel which executes in device";
    }
  }
}

void SessionBasic::AssignParamKey(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // PS embeddingLookup cache check.
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    MS_LOG(EXCEPTION) << "The other parameter can't set ps mode when the embeddingLookup cache is enabled in "
                         "parameter server training mode.";
  }
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      // Assign key for forward kernel EmbeddingLookup.
      // The key will be assigned to embedding table ande Push kernel as well.
      if (AnfAlgo::GetCNodeName(node) == kEmbeddingLookupOpName) {
        size_t embedding_table_idx = 0;
        auto embedding_table = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), embedding_table_idx);
        size_t key = ps::Worker::GetInstance().SetParamKey(embedding_table->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
      } else if (AnfAlgo::GetCNodeName(node) == kPushOpName) {
        auto pull_node = FindPullNode(node, node_list);
        if (!pull_node) {
          MS_LOG(EXCEPTION) << "Assigning parameter key failed: can't find Pull node of the Push node.";
        }

        // Second input of Pull node is the trainable parameter.
        size_t parameter_index = 1;
        auto parameter_node = AnfAlgo::GetInputNode(pull_node->cast<CNodePtr>(), parameter_index);
        size_t key = ps::Worker::GetInstance().SetParamKey(parameter_node->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), pull_node);

        std::string optimizer_name = AnfAlgo::GetNodeAttr<std::string>(node, kAttrOptimizerType);
        ps::Worker::GetInstance().SetKeyOptimId(key, optimizer_name);
      }
    }
  }
}

void SessionBasic::InitPSParamAndOptim(const KernelGraphPtr &kernel_graph,
                                       const std::vector<tensor::TensorPtr> &inputs_const) {
  if (!ps::PSContext::instance()->is_worker()) {
    return;
  }
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto input_nodes = kernel_graph->inputs();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0)) {
      ps::Worker::GetInstance().InitPSParamAndOptim(input_node, tensor);
    }
  }
}
#endif
}  // namespace session
void DumpGraphExeOrder(const std::string &file_name, const std::string &target_dir,
                       const std::vector<CNodePtr> &execution_order) {
  bool status = Common::CreateNotExistDirs(target_dir + "/execution_order/");
  std::string file_path = target_dir + "/execution_order/" + file_name;
  if (!status) {
    MS_LOG(ERROR) << "Failed at CreateNotExistDirs in DumpGraphExeOrder";
    return;
  }
  if (file_path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << file_path << " is too long.";
    return;
  }
  char real_path[PATH_MAX] = {0};
  char *real_path_ret = nullptr;
#if defined(_WIN32) || defined(_WIN64)
  real_path_ret = _fullpath(real_path, file_path.c_str(), PATH_MAX);
#else
  real_path_ret = realpath(file_path.c_str(), real_path);
#endif
  if (nullptr == real_path_ret) {
    MS_LOG(DEBUG) << "dir " << file_path << " does not exit.";
  } else {
    std::string path_string = real_path;
    if (chmod(common::SafeCStr(path_string), S_IRUSR | S_IWUSR) == -1) {
      MS_LOG(ERROR) << "Modify file:" << real_path << " to rw fail.";
      return;
    }
  }

  // write to csv file
  std::ofstream ofs(real_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << real_path << "' failed!";
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
}  // namespace mindspore
