/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/convert_bfloat16.h"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "kernel/common_utils.h"

namespace mindspore::graphkernel {
namespace {
AbstractBasePtr UpdateAbstractDataType(const AbstractBasePtr &orig_abs, TypePtr data_type) {
  if (orig_abs == nullptr) {
    return orig_abs;
  }
  if (orig_abs->isa<abstract::AbstractTensor>()) {
    return std::make_shared<abstract::AbstractTensor>(data_type, orig_abs->GetShape());
  }
  if (orig_abs->isa<abstract::AbstractScalar>()) {
    auto new_abs = orig_abs->Clone();
    new_abs->set_type(data_type);
    return new_abs;
  }
  if (orig_abs->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = orig_abs->cast<abstract::AbstractTuplePtr>()->elements();
    AbstractBasePtrList abstracts(abs_tuple.size());
    for (size_t i = 0; i < abs_tuple.size(); ++i) {
      abstracts[i] = UpdateAbstractDataType(abs_tuple[i], data_type);
    }
    return std::make_shared<abstract::AbstractTuple>(abstracts);
  }
  return orig_abs;
}

void UpdateBuildInfoInputDataType(const AnfNodePtr &node, TypeId orig_type, TypeId new_type) {
  if (node->kernel_info() == nullptr) {
    return;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr) {
    auto inputs_type = build_info->GetAllInputDeviceTypes();
    std::replace_if(
      inputs_type.begin(), inputs_type.end(), [orig_type](TypeId type_id) { return type_id == orig_type; }, new_type);
    build_info->SetInputsDeviceType(inputs_type);
  }
}

void UpdateBuildInfoOutputDataType(const AnfNodePtr &node, TypeId orig_type, TypeId new_type) {
  if (node->kernel_info() == nullptr) {
    return;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr) {
    auto outputs_type = build_info->GetAllOutputDeviceTypes();
    std::replace_if(
      outputs_type.begin(), outputs_type.end(), [orig_type](TypeId type_id) { return type_id == orig_type; }, new_type);
    build_info->SetOutputsDeviceType(outputs_type);
  }
}

AnfNodePtr NewCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, TypeId dst_type) {
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto src_type = cb->GetOutputType(input_node, 0);
  if (dst_type == src_type) {
    return input_node;
  }
  auto type_value = std::make_shared<Int64Imm>(static_cast<int64_t>(dst_type));
  auto type_node = NewValueNode(type_value);
  type_node->set_abstract(type_value->ToAbstract());
  auto cast_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input_node, type_node});
  auto input_abstract = input_node->abstract();
  auto cast_abstract = UpdateAbstractDataType(input_abstract, TypeIdToType(dst_type));
  cast_node->set_abstract(cast_abstract);
  if (cb->IsUseDeviceInfo()) {
    auto input_format = cb->GetOutputFormat(input_node, 0);
    auto input_type = cb->GetOutputType(input_node, 0);
    auto input_object_type = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAbstractObjectType(input_abstract));
    std::string type_node_format = kOpFormat_DEFAULT;
    auto type_node_type = kNumberTypeInt64;
    auto type_node_object_type = kernel::KernelObjectType::SCALAR;
    // set build info for type node
    auto type_kernel_info = std::make_shared<device::KernelInfo>();
    type_node->set_kernel_info(type_kernel_info);
    auto type_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    type_info_builder->SetOutputsFormat(std::vector<std::string>{type_node_format});
    type_info_builder->SetOutputsDeviceType(std::vector<TypeId>{type_node_type});
    type_info_builder->SetOutputsKernelObjectType(std::vector<kernel::KernelObjectType>{type_node_object_type});
    AnfAlgo::SetSelectKernelBuildInfo(type_info_builder->Build(), type_node.get());
    // set build info for cast node
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast_node->set_kernel_info(kernel_info);
    auto info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    info_builder->SetInputsFormat(std::vector<std::string>{input_format, type_node_format});
    info_builder->SetInputsDeviceType(std::vector<TypeId>{input_type, type_node_type});
    info_builder->SetInputsKernelObjectType(
      std::vector<kernel::KernelObjectType>{input_object_type, type_node_object_type});
    info_builder->SetOutputsFormat(std::vector<std::string>{input_format});
    info_builder->SetOutputsDeviceType(std::vector<TypeId>{dst_type});
    info_builder->SetOutputsKernelObjectType(std::vector<kernel::KernelObjectType>{input_object_type});
    AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
  }
  return cast_node;
}

// {prim_name, {inputs_keep_index}}
const HashMap<std::string, std::vector<size_t>> kNeedKeepBF16Ops = {
  {ops::kNameAssign, {kIndex2}}, {ops::kNameMatMul, {kIndex1, kIndex2}}, {ops::kNameBatchMatMul, {kIndex1, kIndex2}}};

inline bool NeedKeepBF16(const CNodePtr &cnode) {
  const auto &prim = GetCNodePrimitive(cnode);
  return prim != nullptr && kNeedKeepBF16Ops.find(prim->name()) != kNeedKeepBF16Ops.end();
}
}  // namespace

AnfNodePtr ConvertBFloat16::GetCastedInput(const AnfNodePtr &input_node, TypeId dst_type,
                                           const FuncGraphPtr &func_graph) {
  auto iter = cast_nodes_.find(input_node);
  if (iter != cast_nodes_.end()) {
    return iter->second;
  }
  cast_nodes_[input_node] = NewCastNode(func_graph, input_node, dst_type);
  return cast_nodes_[input_node];
}

AnfNodePtr ConvertBFloat16::CastTensor(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto *src_data = reinterpret_cast<bfloat16 *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(src_data);
  // create float32 tensor
  auto new_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, tensor->shape());
  MS_EXCEPTION_IF_NULL(new_tensor);
  auto *dst_data = reinterpret_cast<float *>(new_tensor->data_c());
  MS_EXCEPTION_IF_NULL(dst_data);
  for (size_t i = 0; i < tensor->DataSize(); ++i) {
    dst_data[i] = static_cast<float>(src_data[i]);
  }
  // create new value node
  auto new_value_node = NewValueNode(new_tensor);
  new_value_node->set_abstract(new_tensor->ToAbstract());
  if (value_node->kernel_info() != nullptr) {
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(value_node);
    if (build_info != nullptr) {
      // set build info for new value node
      new_value_node->set_kernel_info(std::make_shared<device::KernelInfo>());
      auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(build_info);
      builder->SetOutputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32});
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), new_value_node.get());
    }
  }
  return new_value_node;
}

void ConvertBFloat16::CastInput(const CNodePtr &cnode, size_t input_idx, const FuncGraphPtr &func_graph) {
  input_idx += 1;
  auto input_node = cnode->input(input_idx);
  TypeId target_input_type = kNumberTypeFloat32;
  if (input_node->isa<Parameter>()) {
    auto new_input = GetCastedInput(input_node, target_input_type, func_graph);
    cnode->set_input(input_idx, new_input);
  } else if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    auto new_input = CastTensor(value_node);
    cnode->set_input(input_idx, new_input);
  } else if (IsPrimitiveCNode(input_node, prim::kPrimCast)) {
    // directly link cast's input to current node(because cast bf16 to fp32 needs more intermediate cast)
    auto cast_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cast_node);
    auto new_input = GetCastedInput(cast_node->input(1), target_input_type, func_graph);
    cnode->set_input(input_idx, new_input);
  } else {
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto cur_input_type = cb->GetOutputType(input_node, 0);
    if (cur_input_type != target_input_type) {
      MS_LOG(EXCEPTION) << "For node " << cnode->fullname_with_scope() << ", input[" << input_idx
                        << "]'s data type should already be updated to " << TypeIdToString(target_input_type)
                        << ", but got: " << TypeIdToString(cur_input_type);
    }
  }
}

void ConvertBFloat16::GetKeepBF16Nodes(const FuncGraphPtr &func_graph) {
  keep_bf16_nodes_.clear();
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (NeedKeepBF16(cnode)) {
      // As NeedKeepBF16(cnode), value of GetCNodePrimitive(cnode) is not a nullptr
      auto prim_name = GetCNodePrimitive(cnode)->name();
      for (const auto &input_index : kNeedKeepBF16Ops.at(prim_name)) {
        (void)keep_bf16_nodes_[cnode->input(input_index)].emplace_back(std::make_pair(cnode, input_index));
      }
    } else if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
      auto ret_input = cnode->input(1);
      MS_EXCEPTION_IF_NULL(ret_input);
      if (IsPrimitiveCNode(ret_input, prim::kPrimMakeTuple)) {
        // multiple output
        last_node_ = ret_input->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(last_node_);
        for (size_t i = 1; i < last_node_->size(); ++i) {
          (void)keep_bf16_nodes_[last_node_->input(i)].emplace_back(std::make_pair(last_node_, i));
        }
      } else {
        // single output
        last_node_ = cnode;
        (void)keep_bf16_nodes_[ret_input].emplace_back(std::make_pair(last_node_, 1));
      }
    }
  }
}

bool ConvertBFloat16::Process(const FuncGraphPtr &func_graph) {
  cast_nodes_.clear();
  GetKeepBF16Nodes(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || NeedKeepBF16(cnode)) {
      continue;
    }
    if (cnode == last_node_) {
      break;
    }
    // For cast node, directly update its input data type
    if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      auto orig_input_type = cb->GetInputType(node, 0);
      auto cur_input_type = cb->GetOutputType(cnode->input(1), 0);
      if (cur_input_type != orig_input_type) {
        UpdateBuildInfoInputDataType(node, orig_input_type, cur_input_type);
      }
      continue;
    }
    // For other nodes, add cast for its input and update its abstract and build info
    //   add cast for node's output if node is sub-graph's output
    bool need_update = false;
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(cnode); ++i) {
      auto orig_input_type = cb->GetInputType(cnode, i);
      if (orig_input_type == kNumberTypeBFloat16) {
        need_update = true;
        changed = true;
        CastInput(cnode, i, func_graph);
      }
    }
    if (!need_update) {
      continue;
    }
    auto orig_output_type = cb->GetOutputType(node, 0);
    // update node abstract
    auto new_abstract = UpdateAbstractDataType(node->abstract(), kFloat32);
    node->set_abstract(new_abstract);
    // update node build info
    UpdateBuildInfoInputDataType(node, kNumberTypeBFloat16, kNumberTypeFloat32);
    UpdateBuildInfoOutputDataType(node, kNumberTypeBFloat16, kNumberTypeFloat32);
    // add cast for current node if it is output node
    auto cur_output_type = cb->GetOutputType(node, 0);
    auto iter = keep_bf16_nodes_.find(node);
    if (iter != keep_bf16_nodes_.end() && cur_output_type != orig_output_type) {
      auto new_cast_node = NewCastNode(func_graph, node, orig_output_type);
      for (auto &[user_node, idx] : iter->second) {
        user_node->set_input(idx, new_cast_node);
      }
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ConvertBFloat16::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      changed = Process(sub_graph) || changed;
    }
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
