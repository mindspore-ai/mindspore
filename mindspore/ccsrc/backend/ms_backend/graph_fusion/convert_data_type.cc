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
#include "backend/ms_backend/graph_fusion/convert_data_type.h"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_cse.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "ir/tensor_new.h"
#include "ir/graph_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

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

void UpdateBuildInfoInputDataType(const AnfNodePtr &node, size_t input_idx, TypeId new_type) {
  if (node->kernel_info() == nullptr) {
    return;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr) {
    auto inputs_type = build_info->GetAllInputDeviceTypes();
    if (input_idx >= inputs_type.size()) {
      MS_LOG(EXCEPTION) << "input_idx " << input_idx << " is out of range[0, " << inputs_type.size() << ") for node "
                        << node->fullname_with_scope();
    }
    inputs_type[input_idx] = new_type;
    build_info->SetInputsDeviceType(inputs_type);
  }
}

void UpdateBuildInfoOutputDataType(const AnfNodePtr &node, size_t output_idx, TypeId new_type) {
  if (node->kernel_info() == nullptr) {
    return;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr) {
    auto outputs_type = build_info->GetAllOutputDeviceTypes();
    if (output_idx >= outputs_type.size()) {
      MS_LOG(EXCEPTION) << "output_idx " << output_idx << " is out of range[0, " << outputs_type.size() << ") for node "
                        << node->fullname_with_scope();
    }
    outputs_type[output_idx] = new_type;
    build_info->SetOutputsDeviceType(outputs_type);
  }
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
  MS_EXCEPTION_IF_NULL(func_graph);
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

template <typename S, typename T>
void CastTensorData(void *src, void *dst, size_t size) {
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(dst);
  auto *src_data = reinterpret_cast<S *>(src);
  auto *dst_data = reinterpret_cast<T *>(dst);
  for (size_t i = 0; i < size; ++i) {
    dst_data[i] = static_cast<T>(src_data[i]);
  }
}

const std::map<std::pair<TypeId, TypeId>, std::function<void(void *, void *, size_t)>> cast_map = {
  {{kNumberTypeFloat16, kNumberTypeFloat32},
   [](void *src, void *dst, size_t size) { CastTensorData<float16, float>(src, dst, size); }},
  {{kNumberTypeBFloat16, kNumberTypeFloat32},
   [](void *src, void *dst, size_t size) { CastTensorData<bfloat16, float>(src, dst, size); }},
  {{kNumberTypeInt32, kNumberTypeFloat32},
   [](void *src, void *dst, size_t size) { CastTensorData<int32_t, float>(src, dst, size); }},
  {{kNumberTypeInt32, kNumberTypeFloat16},
   [](void *src, void *dst, size_t size) { CastTensorData<int32_t, float16>(src, dst, size); }},
};

AnfNodePtr CastTensor(const ValueNodePtr &value_node, TypeId src_type, TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto new_tensor = tensor::from_spec(dst_type, tensor->shape(), device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(new_tensor);
  auto it = cast_map.find({src_type, dst_type});
  if (it == cast_map.end()) {
    MS_LOG(EXCEPTION) << "Not supported convert " << TypeIdToString(src_type) << " to " << TypeIdToString(dst_type);
  }
  it->second(tensor->data_c(), new_tensor->data_c(), tensor->DataSize());
  // create new value node
  auto new_value_node = NewValueNode(new_tensor);
  new_value_node->set_abstract(new_tensor->ToAbstract());
  if (value_node->kernel_info() != nullptr) {
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(value_node);
    if (build_info != nullptr) {
      // set build info for new value node
      new_value_node->set_kernel_info(std::make_shared<device::KernelInfo>());
      auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(build_info);
      builder->SetOutputsDeviceType(std::vector<TypeId>{dst_type});
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), new_value_node.get());
    }
  }
  return new_value_node;
}

// {prim_name, {inputs_keep_index}}
const HashMap<std::string, std::vector<size_t>> kNeedKeepBF16Ops = {{ops::kNameAssign, {kIndex2}},
                                                                    {ops::kNameMatMul, {kIndex1, kIndex2}},
                                                                    {ops::kNameBatchMatMul, {kIndex1, kIndex2}},
                                                                    {ops::kNameGroupedMatmul, {kIndex1, kIndex2}},
                                                                    {ops::kNameStridedSlice, {kIndex1}},
                                                                    {ops::kNameSlice, {kIndex1}},
                                                                    {ops::kNameCast, {kIndex1}}};

const HashSet<std::string> kCanKeepBF16Ops = {ops::kNameReshape, kTupleGetItemOpName, ops::kNameBroadcastTo};

inline bool NeedKeepBF16(const CNodePtr &cnode) {
  const auto &prim = GetCNodePrimitive(cnode);
  return prim != nullptr && kNeedKeepBF16Ops.find(prim->name()) != kNeedKeepBF16Ops.end();
}

inline bool CanKeepBF16(const CNodePtr &cnode) {
  const auto &prim = GetCNodePrimitive(cnode);
  return prim != nullptr && kCanKeepBF16Ops.find(prim->name()) != kCanKeepBF16Ops.end();
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

void ConvertBFloat16::CastInput(const CNodePtr &cnode, size_t input_idx, const FuncGraphPtr &func_graph) {
  input_idx += 1;
  auto input_node = cnode->input(input_idx);
  TypeId target_input_type = kNumberTypeFloat32;
  if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    auto new_input = CastTensor(value_node, kNumberTypeBFloat16, target_input_type);
    cnode->set_input(input_idx, new_input);
  } else if (IsPrimitiveCNode(input_node, prim::kPrimCast)) {
    // directly link cast's input to current node(because cast bf16 to fp32 needs more intermediate cast)
    auto cast_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cast_node);
    auto new_input = GetCastedInput(cast_node->input(1), target_input_type, func_graph);
    cnode->set_input(input_idx, new_input);
  } else {
    auto new_input = GetCastedInput(input_node, target_input_type, func_graph);
    cnode->set_input(input_idx, new_input);
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
      auto cnode_prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(cnode_prim);
      auto prim_name = cnode_prim->name();
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
    // For other nodes, add cast for its input and update its abstract and build info
    // add cast for node's output if node is sub-graph's output
    bool need_update = false;
    bool can_keep_bf16 = CanKeepBF16(cnode);
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(cnode); ++i) {
      auto ori_input_type = cb->GetInputType(cnode, i);
      if (ori_input_type != kNumberTypeBFloat16) {
        continue;
      }
      if (can_keep_bf16) {
        auto input_node = cnode->input(i + 1);
        if (input_node->isa<ValueNode>()) {
          continue;
        }
        auto cur_input_type = cb->GetOutputType(input_node, 0);
        if (cur_input_type != ori_input_type) {
          need_update = changed = true;
        }
      } else {
        need_update = changed = true;
        CastInput(cnode, i, func_graph);
      }
    }
    if (!need_update) {
      continue;
    }
    auto orig_output_type = cb->GetOutputType(node, 0);
    // update node abstract
    if (orig_output_type == kNumberTypeBFloat16) {
      auto new_abstract = UpdateAbstractDataType(node->abstract(), kFloat32);
      node->set_abstract(new_abstract);
    }
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

void ConvertMixType::CastInput(const CNodePtr &cnode, size_t input_idx, TypeId cur_type, TypeId target_type) {
  if (cur_type == target_type) {
    return;
  }
  MS_LOG(DEBUG) << "node: " << cnode->fullname_with_scope() << " input_idx: " << input_idx << " "
                << TypeIdToString(cur_type) << " --> " << TypeIdToString(target_type);
  input_idx += 1;
  auto input_node = cnode->input(input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  auto new_input_node = input_node->isa<ValueNode>()
                          ? CastTensor(input_node->cast<ValueNodePtr>(), cur_type, target_type)
                          : NewCastNode(cnode->func_graph(), input_node, target_type);
  cnode->set_input(input_idx, new_input_node);
  // update node input build info
  UpdateBuildInfoInputDataType(cnode, input_idx - 1, target_type);
}

void ConvertMixType::CastOutput(const CNodePtr &cnode, size_t output_idx, TypeId cur_type, TypeId target_type) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Multiple outputs not supported now! node: " << cnode->fullname_with_scope();
  }
  if (cur_type == target_type) {
    return;
  }
  MS_LOG(DEBUG) << "node: " << cnode->fullname_with_scope() << " output_idx: " << output_idx << " "
                << TypeIdToString(cur_type) << " --> " << TypeIdToString(target_type);
  // update node output abstract
  cnode->set_abstract(UpdateAbstractDataType(cnode->abstract(), TypeIdToType(target_type)));
  // update node output build info
  UpdateBuildInfoOutputDataType(cnode, output_idx, target_type);
  // update node users
  auto new_node = NewCastNode(cnode->func_graph(), cnode, cur_type);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  const auto &users = mng->node_users()[cnode];
  for (const auto &user : users) {
    auto user_node = user.first;
    size_t user_index = IntToSize(user.second);
    if (user_node != nullptr && user_node->isa<CNode>()) {
      auto user_cnode = user_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_cnode);
      user_cnode->set_input(user_index, new_node);
    }
  }
}

bool ConvertMixType::ProcessBinary(const CNodePtr &cnode) {
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto x0_type = cb->GetInputType(cnode, 0);
  auto x1_type = cb->GetInputType(cnode, 1);
  auto out_type = cb->GetOutputType(cnode, 0);
  if (x0_type == out_type && x1_type == out_type) {
    return false;
  }
  auto compute_type = ops::PromoteType(x0_type, x1_type, cnode->fullname_with_scope());
  if (compute_type == kNumberTypeBFloat16) {
    compute_type = kNumberTypeFloat32;
  }
  CastInput(cnode, 0, x0_type, compute_type);
  CastInput(cnode, 1, x1_type, compute_type);
  CastOutput(cnode, 0, out_type, compute_type);
  return true;
}

bool ConvertMixType::Process(const FuncGraphPtr &func_graph) {
  using ProcessFunc = bool (ConvertMixType::*)(const CNodePtr &);
  static const std::map<std::string, ProcessFunc> funcs = {
    {ops::kNameMul, &ConvertMixType::ProcessBinary},
  };

  bool changed = false;
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto op_name = AnfUtils::GetCNodeName(cnode);
      auto it = funcs.find(op_name);
      if (it != funcs.end()) {
        changed = (this->*(it->second))(cnode) || changed;
      }
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ConvertMixType::Run(const FuncGraphPtr &func_graph) {
  auto changed = ConvertDataType::Run(func_graph);
  if (changed) {
    changed = std::make_shared<GraphKernelCSE>()->Run(func_graph) || changed;
  }
  return changed;
}

bool ConvertDataType::Run(const FuncGraphPtr &func_graph) {
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
