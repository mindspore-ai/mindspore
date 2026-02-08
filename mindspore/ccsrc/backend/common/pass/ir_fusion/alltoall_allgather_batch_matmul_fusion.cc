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
#include "backend/common/pass/ir_fusion/alltoall_allgather_batch_matmul_fusion.h"
#include <set>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <functional>
#include "backend/common/pass/ir_fusion/mc2_fusion.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "primitive/math_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::opt {
namespace {
const PrimitiveSet valid_act_prim_list = {prim::kPrimGeLU, prim::kPrimSiLU, prim::kPrimReLU, prim::kPrimFastGeLU};
std::unordered_map<std::string, int64_t> act_type_map = {{prim::kPrimGeLU->name(), 1},
                                                         {prim::kPrimSiLU->name(), 2},
                                                         {prim::kPrimReLU->name(), 3},
                                                         {prim::kPrimFastGeLU->name(), 4}};
constexpr int64_t kSplitTwo = 2;
constexpr int64_t kSplitFour = 4;
constexpr int64_t kSplitEight = 8;
constexpr int64_t kSplitSixteen = 16;
using AnfNodeIndex = std::pair<AnfNodePtr, int>;
using AnfNodeIndexList = std::vector<AnfNodeIndex>;

ShapeVector GetShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto base_shape = abstract->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);

  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  }
  return {};
}

namespace {
bool EnableLccl() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b" && ascend_soc_version != "ascend910_93") {
    return false;
  }
  auto enable_infer_boost = MsContext::GetInstance()->IsEnableInferBoost();
  if (enable_infer_boost) {
    static bool disable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "off";
    if (disable_lccl) {
      return false;
    }
    return true;
  } else {
    static bool enable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "on";
    if (enable_lccl) {
      MS_LOG(EXCEPTION)
        << "Currently, the LCCL communication library is only supported in the inference boost scenario. "
           "Please remove the environment variable MS_ENABLE_LCCL=on.";
    }
    return false;
  }
}
}  // namespace

bool IsKbkAclnnMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  bool is_k_by_k_mode = ms_context->IsKByKExecutorMode();
  bool enable_lccl = EnableLccl();
  return is_k_by_k_mode && !enable_lccl;
}

bool IsAscendVerison93() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  return soc_version == "ascend910_93";
}

ValuePtr GetAttrFromCNodePrimitive(const AnfNodePtr &node, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto attr = prim->GetAttr(attr_name);
  if (attr == nullptr) {
    MS_LOG(EXCEPTION) << "Get attribute " << attr_name << " from node " << node->DebugString() << " failed.";
  }
  return attr;
}

AnfNodeIndex SkipNodes(AnfNodeIndex user, const FuncGraphManagerPtr &manager, const PrimitiveSet &skip_prim_set) {
  while (IsOneOfPrimitiveCNode(user.first, skip_prim_set)) {
    const auto &next_users = manager->node_users()[user.first];
    if (next_users.empty()) {
      break;
    }
    user = next_users.front();
  }
  return user;
}

bool GetExpectedUserAndOtherUserList(const AnfNodePtr &node, const PrimitiveSet &expect_prim_set,
                                     const PrimitiveSet &skip_prim_set, bool allow_multi_user,
                                     AnfNodeIndex *expect_user, AnfNodeIndexList *other_user_list) {
  MS_EXCEPTION_IF_NULL(node);
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_user_list = manager->node_users()[node];
  if (!allow_multi_user && node_user_list.size() > 1) {
    return false;
  }

  other_user_list->clear();
  bool found_expect_user = false;
  for (const auto &user : node_user_list) {
    if (found_expect_user) {
      other_user_list->push_back(user);
      continue;
    }
    auto final_user = SkipNodes(user, manager, skip_prim_set);
    if (IsOneOfPrimitiveCNode(final_user.first, expect_prim_set)) {
      *expect_user = final_user;
      found_expect_user = true;
    } else {
      other_user_list->push_back(user);
    }
  }
  return found_expect_user;
}

void GetAllGatherLastNode(const CNodePtr &allgather_cnode, CNodePtr *last_cnode) {
  MS_EXCEPTION_IF_NULL(allgather_cnode);
  if (!IsPrimitiveCNode(allgather_cnode, prim::kPrimAllGather)) {
    MS_LOG(EXCEPTION) << "Input cnode must be AllGather, but got node " << allgather_cnode->fullname_with_scope();
  }
  *last_cnode = allgather_cnode;

  auto func_graph = allgather_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // After the fusion, there is no output of all_gather, only the concat output after all_gather, so all_gather cannot
  // have other users
  const auto &user_list = manager->node_users()[allgather_cnode];
  if (user_list.size() > 1) {
    return;
  }
  const std::vector<PrimitivePtr> pattern = {prim::kPrimSplit, prim::kPrimTupleGetItem, prim::kPrimMakeTuple,
                                             prim::kPrimConcat};
  AnfNodePtr cur_node = allgather_cnode;
  for (const auto &expect_prim : pattern) {
    auto users = manager->node_users()[cur_node];
    if (users.empty()) {
      return;
    }

    cur_node = users.front().first;
    MS_EXCEPTION_IF_NULL(cur_node);
    if (!IsPrimitiveCNode(cur_node, expect_prim)) {
      return;
    }
  }
  *last_cnode = cur_node->cast<CNodePtr>();
}

std::vector<AnfNodePtr> GetNodeUsers(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  const auto &users = manager->node_users()[node];
  std::vector<AnfNodePtr> user_nodes;
  for (const auto &user : users) {
    MS_EXCEPTION_IF_NULL(user.first);
    user_nodes.push_back(user.first);
  }
  return user_nodes;
}

CNodePtr CreateTupleGetItemCNode(const FuncGraphPtr &func_graph, const CNodePtr &pre_cnode, int64_t index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto tuple_get_item_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), pre_cnode, NewValueNode(MakeValue<int64_t>(index))});
  auto shape = common::AnfAlgo::GetOutputInferShape(pre_cnode, index);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(pre_cnode, index);
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {shape}, tuple_get_item_cnode.get());
  return tuple_get_item_cnode;
}

CNodePtr CreateReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const ShapeVector &to_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto from_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  // Calculate the total number of elements in the input shape and target shape.
  auto from_elements = std::accumulate(from_shape.begin(), from_shape.end(), int64_t{1}, std::multiplies<int64_t>());
  auto to_elements = std::accumulate(to_shape.begin(), to_shape.end(), int64_t{1}, std::multiplies<int64_t>());
  // Check whether the total number of elements is consistent.
  if (from_elements != to_elements) {
    MS_LOG(EXCEPTION) << "Failed to insert reshape after " << input_node->fullname_with_scope()
                      << ". The total number of elements in from_shape (" << from_elements
                      << ") does not match to_shape (" << to_elements << ").";
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto to_shape_node = kernel_graph->NewValueNode(MakeValue<std::vector<int64_t>>(to_shape));
  kernel_graph->AddValueNodeToGraph(to_shape_node);

  auto reshape_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimReshape), input_node, to_shape_node});
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, kIndex0);
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {to_shape}, reshape_cnode.get());
  return reshape_cnode;
}

// If needed, insert the Reshape action before the specified input for the target_node.
// If the input shape already matches the desired shape, no Reshape is inserted and the original input node is returned
// directly. Returns the inserted Reshape node; If not plugged in, the original input node is returned.
AnfNodePtr InsertReshapeCNodeBefore(const FuncGraphPtr func_graph, const CNodePtr target_node,
                                    const ShapeVector &to_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_node);
  auto input_node = target_node->input(kIndex1);
  auto from_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  if (from_shape == to_shape) {
    MS_LOG(DEBUG) << "The from_shape is equal to to_shape, skip it.";
    return input_node;
  }
  auto reshape_cnode = CreateReshapeCNode(func_graph, input_node, to_shape);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(target_node, kIndex1, reshape_cnode);
  return reshape_cnode;
}

void PostProcessForOutput(const FuncGraphPtr &func_graph, const CNodePtr &fusion_cnode, int64_t index,
                          const CNodePtr origin_cnode, const AnfNodeIndexList &user_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(fusion_cnode);
  MS_EXCEPTION_IF_NULL(origin_cnode);

  auto output_cnode = CreateTupleGetItemCNode(func_graph, fusion_cnode, index);
  MS_EXCEPTION_IF_NULL(output_cnode);
  auto cur_shape = common::AnfAlgo::GetOutputInferShape(output_cnode, kIndex0);
  auto ori_shape = common::AnfAlgo::GetOutputInferShape(origin_cnode, kIndex0);
  if (cur_shape != ori_shape) {
    output_cnode = CreateReshapeCNode(func_graph, output_cnode, ori_shape);
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto user : user_list) {
    manager->SetEdge(user.first, user.second, output_cnode);
  }
}
}  // namespace

void AllToAllAllGatherBatchMatMulFusion::SetOutputTypeAndShapeForAllToAllAllGatherBatchMatMul(
  const AnfNodePtr &alltoall_allgather_batch_matmul_node, bool output_y2_flag, bool output_y3_flag) {
  MS_EXCEPTION_IF_NULL(alltoall_allgather_batch_matmul_node);

  auto batch_matmul_dtype = common::AnfAlgo::GetOutputInferDataType(batch_matmul_cnode_, kIndex0);
  auto batch_matmul_shape = common::AnfAlgo::GetOutputInferShape(batch_matmul_cnode_, kIndex0);

  auto all_gather_last_dtype = common::AnfAlgo::GetOutputInferDataType(allgather_last_cnode_, kIndex0);
  auto all_gather_last_shape = common::AnfAlgo::GetOutputInferShape(allgather_last_cnode_, kIndex0);

  auto AdjustShaperTo3D = [](ShapeVector shape) -> ShapeVector {
    if (shape.size() > kSizeThree) {
      int64_t first_dim_product = 1;
      for (size_t i = 0; i < shape.size() - 2; ++i) {
        first_dim_product *= shape[i];
      }

      return {first_dim_product, shape[shape.size() - 2], shape[shape.size() - 1]};
    }
    return shape;
  };

  auto shape_3d_batch_matmul = AdjustShaperTo3D(batch_matmul_shape);

  ShapeVector shape_3d_all_gather_last = {};

  std::vector<TypeId> output_types;
  std::vector<abstract::BaseShapePtr> output_shapes;

  // y1_out: If there is an activation function, it is the output of the activation function. If there is no activation
  // function, it is the output of batchmatmul.
  if (act_cnode_ != nullptr) {
    auto act_dtype = common::AnfAlgo::GetOutputInferDataType(act_cnode_, kIndex0);
    auto act_shape = common::AnfAlgo::GetOutputInferShape(act_cnode_, kIndex0);
    auto shape_3d_act = AdjustShaperTo3D(act_shape);
    shape_3d_all_gather_last = shape_3d_act;
    output_types.push_back(act_dtype);
    output_shapes.push_back(std::make_shared<abstract::Shape>(shape_3d_act));
  } else {
    output_types.push_back(batch_matmul_dtype);
    output_shapes.push_back(std::make_shared<abstract::Shape>(shape_3d_batch_matmul));
    shape_3d_all_gather_last = shape_3d_batch_matmul;
  }

  shape_3d_all_gather_last[kIndex2] = all_gather_last_shape[all_gather_last_shape.size() - 1];

  // y2_out: The output of allgather_last.
  if (output_y2_flag) {
    output_types.push_back(all_gather_last_dtype);
    output_shapes.push_back(std::make_shared<abstract::Shape>(shape_3d_all_gather_last));
  }

  // y3_out: The output of batchmatmul.
  if (output_y3_flag) {
    output_types.push_back(batch_matmul_dtype);
    output_shapes.push_back(std::make_shared<abstract::Shape>(shape_3d_batch_matmul));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(output_types, output_shapes, alltoall_allgather_batch_matmul_node.get());
}

// Create node and set abstract by call op's infer
CNodePtr AllToAllAllGatherBatchMatMulFusion::PreProcessAndCreateAllToAllAllGatherBatchMatMulCNode(
  const FuncGraphPtr func_graph, bool output_y2_flag, bool output_y3_flag) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(alltoall_cnode_);
  MS_EXCEPTION_IF_NULL(allgather_cnode_);
  MS_EXCEPTION_IF_NULL(batch_matmul_cnode_);

  auto prim = prim::kPrimAlltoAllAllGatherBatchMatMul->Clone();
  MS_EXCEPTION_IF_NULL(prim);

  auto group_ep_attr = GetAttrFromCNodePrimitive(alltoall_cnode_, kAttrGroup);
  MS_CHECK_TRUE_RET(group_ep_attr != nullptr, nullptr);
  auto group_ep = GetValue<std::string>(group_ep_attr);
  constexpr int64_t kMaxGroupEpAndTpSize = 128;
  MS_CHECK_TRUE_RET(!group_ep.empty() && group_ep.size() < kMaxGroupEpAndTpSize, nullptr);

  auto group_tp_attr = GetAttrFromCNodePrimitive(allgather_cnode_, kAttrGroup);
  MS_CHECK_TRUE_RET(group_tp_attr != nullptr, nullptr);
  auto group_tp = GetValue<std::string>(group_tp_attr);
  MS_CHECK_TRUE_RET(!group_tp.empty() && group_tp.size() < kMaxGroupEpAndTpSize, nullptr);

  MS_CHECK_TRUE_RET(ep_world_size_ == kSplitTwo || ep_world_size_ == kSplitFour || ep_world_size_ == kSplitEight ||
                      ep_world_size_ == kSplitSixteen,
                    nullptr);
  MS_CHECK_TRUE_RET(tp_world_size_ == kSplitTwo || tp_world_size_ == kSplitFour || tp_world_size_ == kSplitEight ||
                      tp_world_size_ == kSplitSixteen,
                    nullptr);

  MS_CHECK_TRUE_RET(x_shard_type_ == 1, nullptr);

  prim->AddAttr(kAttrGroupEp, group_ep_attr);
  prim->AddAttr(kAttrGroupTp, group_tp_attr);
  prim->AddAttr(kAttrEpWorldSize, MakeValue<int64_t>(ep_world_size_));
  prim->AddAttr(kAttrTpWorldSize, MakeValue<int64_t>(tp_world_size_));
  prim->AddAttr(kAttrXShardType, MakeValue<int64_t>(x_shard_type_));

  int64_t act_type = 0;
  if (with_act_calc_) {
    MS_EXCEPTION_IF_NULL(act_cnode_);
    auto act_prim = GetCNodePrimitive(act_cnode_);
    MS_EXCEPTION_IF_NULL(act_prim);
    auto act_prim_name = act_prim->name();
    if (act_type_map.find(act_prim_name) == act_type_map.end()) {
      MS_LOG(DEBUG) << "Unsupported activation op: " << act_cnode_->fullname_with_scope();
      return nullptr;
    }
    act_type = act_type_map[act_prim_name];
  }
  prim->AddAttr(kAttrActType, MakeValue<int64_t>(act_type));

  auto transpose_weight_value_node = batch_matmul_cnode_->input(kIndex4)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(transpose_weight_value_node);
  auto transpose_weight = false;
  if (transpose_weight_value_node->value()->isa<BoolImm>()) {
    transpose_weight = GetValue<bool>(transpose_weight_value_node->value());
  } else {
    MS_LOG(EXCEPTION) << "The transpose_weight value node is not bool imm";
  }
  prim->AddAttr(kAttrTransposeWeight, MakeValue<bool>(transpose_weight));
  prim->AddAttr(kAttrOutputY2Flag, MakeValue<bool>(output_y2_flag));
  prim->AddAttr(kAttrOutputY3Flag, MakeValue<bool>(output_y3_flag));

  auto x = alltoall_cnode_->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto weight = batch_matmul_cnode_->input(kIndex2);
  MS_EXCEPTION_IF_NULL(weight);
  AnfNodePtrList inputs = {NewValueNode(prim), x, weight};
  if (with_bias_add_) {
    MS_EXCEPTION_IF_NULL(bias_add_cnode_);
    auto bias_input = bias_add_cnode_->input(kIndex2);
    MS_EXCEPTION_IF_NULL(bias_input);
    inputs.push_back(bias_input);
  }
  auto alltoall_allgather_batch_matmul_cnode = func_graph->NewCNode(inputs);

  // Insert reshape for x and weight
  ShapeVector expect_x_shape = {expert_size_, capacity_size_, hidden_size_};
  auto x_reshape = InsertReshapeCNodeBefore(func_graph, alltoall_allgather_batch_matmul_cnode, expect_x_shape);
  MS_EXCEPTION_IF_NULL(x_reshape);

  constexpr int64_t kMinExpertSize = 2;
  constexpr int64_t kMaxExpertSize = 2048;
  MS_CHECK_TRUE_RET(expert_size_ >= kMinExpertSize && expert_size_ <= kMaxExpertSize, nullptr);
  MS_CHECK_TRUE_RET(expert_size_ % ep_world_size_ == 0, nullptr);
  constexpr int64_t kMinHiddenSize = 1;
  constexpr int64_t kMaxHiddenSize = 65535;
  MS_CHECK_TRUE_RET(hidden_size_ >= kMinHiddenSize && hidden_size_ <= kMaxHiddenSize, nullptr);

  ShapeVector origin_weight_shape = common::AnfAlgo::GetOutputInferShape(weight, kIndex0);
  if (origin_weight_shape.size() != kSizeThree) {
    MS_LOG(EXCEPTION) << "The weight shape size is not 3";
  }

  auto first_dim_size = origin_weight_shape[kIndex0];
  constexpr int64_t kMinDimSize = 1;
  MS_CHECK_TRUE_RET(first_dim_size >= kMinDimSize, nullptr);
  MS_CHECK_TRUE_RET(expert_size_ % first_dim_size == 0, nullptr);

  auto third_dim_size = origin_weight_shape[kIndex2];

  constexpr int64_t kMinThirdDimSize = 1;
  constexpr int64_t kMaxThirdDimSize = 65535;
  MS_CHECK_TRUE_RET(third_dim_size >= kMinThirdDimSize && third_dim_size <= kMaxThirdDimSize, nullptr);

  SetOutputTypeAndShapeForAllToAllAllGatherBatchMatMul(alltoall_allgather_batch_matmul_cnode, output_y2_flag,
                                                       output_y3_flag);
  return alltoall_allgather_batch_matmul_cnode;
}

void AllToAllAllGatherBatchMatMulFusion::InitAttr() {
  reshape_before_batch_matmul_cnode_ = nullptr;
  alltoall_cnode_ = nullptr;
  allgather_cnode_ = nullptr;
  batch_matmul_cnode_ = nullptr;
  bias_add_cnode_ = nullptr;
  act_cnode_ = nullptr;
  last_cnode_ = nullptr;
  with_bias_add_ = false;
  with_act_calc_ = false;
  split_dim_ = -1;
  concat_dim_ = -1;
  expert_size_ = -1;
  capacity_size_ = -1;
  hidden_size_ = -1;
  ep_world_size_ = -1;
  tp_world_size_ = -1;
  x_shard_type_ = 1;
  is_grad_ = false;
}

bool AllToAllAllGatherBatchMatMulFusion::IsValidAllToAll(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimAlltoAll) && !IsPrimitiveCNode(node, prim::kPrimAllToAll)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (cnode->inputs().size() < kSizeTwo) {
    return false;
  }
  auto all_to_all_input_node = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(all_to_all_input_node);

  auto all_to_all_input_shape = GetShape(all_to_all_input_node);
  if (all_to_all_input_shape.empty()) {
    return false;
  }
  int64_t all_to_all_input_dim = static_cast<int64_t>(all_to_all_input_shape.size());

  auto split_dim_attr = GetAttrFromCNodePrimitive(cnode, kAttrSplitDim);
  auto concat_dim_attr = GetAttrFromCNodePrimitive(cnode, kAttrConcatDim);

  split_dim_ = GetValue<int64_t>(split_dim_attr);
  concat_dim_ = GetValue<int64_t>(concat_dim_attr);

  if (all_to_all_input_dim < kSplitFour) {
    return false;
  }

  int64_t expect_split_dim = all_to_all_input_dim - 4;
  int64_t expect_concat_dim = all_to_all_input_dim - 3;

  // The input of all_to_all may be four-dimensional or five-dimensional. When it is four-dimensional, the 0th and 1st
  // dimensions are split and contact respectively. When the input is five-dimensional, the 1st and 2nd dimensions are
  // split and contact respectively.
  return (split_dim_ == expect_split_dim && concat_dim_ == expect_concat_dim);
}

void AllToAllAllGatherBatchMatMulFusion::InferInputDimByAllToAllCNode() {
  MS_EXCEPTION_IF_NULL(alltoall_cnode_);

  auto input_node = alltoall_cnode_->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_node);

  auto input_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  if (input_shape.size() < kSizeThree) {
    MS_LOG(EXCEPTION) << "The input of alltoall should be four-dimensional or five-dimensional but got "
                      << input_shape.size();
  }

  ep_world_size_ = GetValue<int64_t>(GetAttrFromCNodePrimitive(alltoall_cnode_, kAttrSplitCount));

  // expert_size_ = input_shape[0] * input_shape[1] * ... * input_shape[-3]
  expert_size_ = std::accumulate(input_shape.begin(), input_shape.end() - kSizeTwo, static_cast<int64_t>(1),
                                 std::multiplies<int64_t>());

  // capacity_size_ = input_shape[-2]
  capacity_size_ = input_shape[input_shape.size() - 2];

  // capacity_size_ = input_shape[-1]
  hidden_size_ = input_shape.back();
}

void AllToAllAllGatherBatchMatMulFusion::FindFirstNonReshapeAllToAllUser(const AnfNodePtr &node, bool *is_grad_) {
  MS_EXCEPTION_IF_NULL(node);
  auto manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);

  PrimitiveSet skip_prim_set = {prim::kPrimReshape, prim::kPrimStridedSlice, prim::kPrimStridedSliceGrad};
  const auto &node_users = manager->node_users()[node];
  for (const auto &user : node_users) {
    auto current_user = user;
    while (current_user.first) {
      if (IsOneOfPrimitiveCNode(current_user.first, skip_prim_set)) {
        if (IsPrimitiveCNode(current_user.first, prim::kPrimStridedSliceGrad)) {
          *is_grad_ = true;
        }

        current_user = manager->node_users()[current_user.first].empty()
                         ? AnfNodeIndex()
                         : manager->node_users()[current_user.first].front();
      } else {
        return;
      }
    }
  }
}

// Replaces nodes in the calculation graph.
void AllToAllAllGatherBatchMatMulFusion::ReplaceGraph(
  const FuncGraphPtr &func_graph, const CNodePtr &fusion_cnode,
  const AnfNodeIndexList &reshape_before_batch_matmul_user_other_users, const AnfNodeIndexList &bias_add_other_users,
  bool output_y2_flag, bool output_y3_flag) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Process the y1 output.
  auto last_cnode_users = manager->node_users()[last_cnode_];
  AnfNodeIndexList last_cnode_user_list(last_cnode_users.begin(), last_cnode_users.end());
  PostProcessForOutput(func_graph, fusion_cnode, kIndex0, last_cnode_, last_cnode_user_list);

  // Process the y2 output.
  if (output_y2_flag) {
    PostProcessForOutput(func_graph, fusion_cnode, kIndex1, reshape_before_batch_matmul_cnode_,
                         reshape_before_batch_matmul_user_other_users);
  }

  // Process the y3 output.
  if (output_y3_flag) {
    PostProcessForOutput(func_graph, fusion_cnode, kIndex2, bias_add_cnode_, bias_add_other_users);
  }
}

CNodePtr AllToAllAllGatherBatchMatMulFusion::FindReshapeBeforeBatchMatMul(const FuncGraphPtr &func_graph,
                                                                          const AnfNodePtr &node) {
  // Get the list of users for the input node.
  auto node_users = GetNodeUsers(func_graph, node);

  // Iterate over the users of the current node.
  for (const auto &user_node : node_users) {
    // Check if the user node is a Reshape node.
    if (IsPrimitiveCNode(user_node, prim::kPrimReshape)) {
      auto reshape_cnode = user_node->cast<CNodePtr>();
      auto reshape_node_users = GetNodeUsers(func_graph, reshape_cnode);

      // Check if there is a single or multiple user nodes, and if the user is a BatchMatMul node.
      for (const auto &reshape_node_user : reshape_node_users) {
        if (IsPrimitiveCNode(reshape_node_user, prim::kPrimBatchMatMul)) {
          return reshape_cnode;
        }
      }

      // If Reshape has multiple users but none lead to BatchMatMul, return nullptr as it can't be uniquely determined.
      if (reshape_node_users.size() > 1) {
        return nullptr;
      }

      // Recursively check for BatchMatMul if conditions are not satisfied.
      return FindReshapeBeforeBatchMatMul(func_graph, user_node);
    }

    // If the user node is a StridedSlice, continue searching recursively.
    if (!is_grad_ && IsPrimitiveCNode(user_node, prim::kPrimStridedSlice)) {
      return FindReshapeBeforeBatchMatMul(func_graph, user_node);
    }

    if (is_grad_ &&
        (IsPrimitiveCNode(user_node, prim::kPrimTranspose) || IsPrimitiveCNode(user_node, prim::kPrimTransposeView))) {
      return FindReshapeBeforeBatchMatMul(func_graph, user_node);
    }
  }

  // If no matching Reshape node is found, return nullptr.
  return nullptr;
}

AnfNodePtr AllToAllAllGatherBatchMatMulFusion::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // 1. Match fusion pattern and get attribute
  // 1.1 AllToAll && The first input of node AllToAll is used to obtain the required E, C, H, etc. of the fusion
  // operator.
  if (!IsValidAllToAll(node)) {
    return nullptr;
  }

  MS_LOG(DEBUG) << "Try to fusion for node: " << node->fullname_with_scope();
  InitAttr();
  alltoall_cnode_ = node->cast<CNodePtr>();

  auto all_to_all_users = GetNodeUsers(func_graph, node);
  if (all_to_all_users.empty()) {
    return nullptr;
  }
  FindFirstNonReshapeAllToAllUser(node, &is_grad_);

  // Get the original E, C, H, etc. values of the first input of AllTOAll before fusion
  InferInputDimByAllToAllCNode();

  // 1 - If all_to_all is followed by StrideSlice, check whether it is AllToAll -> StrideSlice_useless -> AllGather ->
  // Split -> multi. TupleGetItem -> MakeTuple -> Concat -> StrideSlice_useless -> Reshape (with other_users) ->
  // BatchMatMul

  // 2 - If all_to_all is followed by StrideSliceGrad, check whether it is AllToAll -> StrideSliceGrad ->Transpose ->
  // AllGather ->. Transpose -> Reshape(with other_users) -> BatchMatMul
  AnfNodeIndex allgather_user;
  AnfNodeIndexList alltoall_other_users;
  if (!is_grad_) {
    if (!GetExpectedUserAndOtherUserList(
          alltoall_cnode_, {prim::kPrimAllGather},
          {prim::kPrimStridedSlice, prim::kPrimReshape, prim::kPrimSplit, prim::kPrimConcat, prim::kPrimTupleGetItem},
          false, &allgather_user, &alltoall_other_users)) {
      MS_LOG(DEBUG) << "Cannot find expected AllGather user.";
      return nullptr;
    }
    // allgather_cnode_ = AllGather
    allgather_cnode_ = allgather_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(allgather_cnode_ != nullptr, nullptr);
    GetAllGatherLastNode(allgather_cnode_, &allgather_last_cnode_);

    // allgather_last_cnode_ = concat
    MS_CHECK_TRUE_RET(allgather_last_cnode_ != nullptr, nullptr);
  } else {
    if (!GetExpectedUserAndOtherUserList(
          alltoall_cnode_, {prim::kPrimAllGather},
          {prim::kPrimReshape, prim::kPrimStridedSliceGrad, prim::kPrimTranspose, prim::kPrimTransposeView}, false,
          &allgather_user, &alltoall_other_users)) {
      MS_LOG(DEBUG) << "Cannot find expected AllGather user.";
      return nullptr;
    }
    allgather_cnode_ = allgather_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(allgather_cnode_ != nullptr, nullptr);
    allgather_last_cnode_ = allgather_cnode_;
  }
  tp_world_size_ = GetValue<int64_t>(GetAttrFromCNodePrimitive(allgather_cnode_, kAttrRankSize));

  // reshape_before_batch_matmul_cnode_ = reshape -> batchmatmul
  reshape_before_batch_matmul_cnode_ = FindReshapeBeforeBatchMatMul(func_graph, allgather_last_cnode_);

  MS_CHECK_TRUE_RET(reshape_before_batch_matmul_cnode_ != nullptr, nullptr);

  // 1.3 BatchMatMul and bias add
  // batch_matmul_user = batchmatmul
  AnfNodeIndex batch_matmul_user;

  // reshape_before_batch_matmul_user_other_users = reshape(other_users) -> batchmatmul
  AnfNodeIndexList reshape_before_batch_matmul_user_other_users;
  if (!GetExpectedUserAndOtherUserList(reshape_before_batch_matmul_cnode_, {prim::kPrimBatchMatMul}, {}, true,
                                       &batch_matmul_user, &reshape_before_batch_matmul_user_other_users)) {
    MS_LOG(DEBUG) << "Cannot find expected BatchMatMul user.";
    return nullptr;
  }

  batch_matmul_cnode_ = batch_matmul_user.first->cast<CNodePtr>();
  AnfNodeIndex bias_add_user;
  AnfNodeIndexList batch_matmul_other_users;
  with_bias_add_ = GetExpectedUserAndOtherUserList(batch_matmul_cnode_, {prim::kPrimAdd}, {prim::kPrimReshape}, false,
                                                   &bias_add_user, &batch_matmul_other_users);
  bias_add_cnode_ = batch_matmul_cnode_;
  if (with_bias_add_) {
    bias_add_cnode_ = bias_add_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(bias_add_cnode_);
  }
  last_cnode_ = bias_add_cnode_;

  // 1.4 Activation
  AnfNodeIndex act_user;
  AnfNodeIndexList bias_add_other_users;
  with_act_calc_ = GetExpectedUserAndOtherUserList(bias_add_cnode_, valid_act_prim_list, {prim::kPrimReshape}, true,
                                                   &act_user, &bias_add_other_users);
  if (with_act_calc_) {
    act_cnode_ = act_user.first->cast<CNodePtr>();
    last_cnode_ = act_cnode_;
    MS_EXCEPTION_IF_NULL(act_cnode_);
  }

  // 2. Create Fusion Node
  // The output of y2 is the output of all_gather. If the reshape after all_gather and before batchmatmul has other
  // users, then the output of y2 is required.
  auto output_y2_flag = !reshape_before_batch_matmul_user_other_users.empty();

  // When there is a specific activation function, there should be an output of y3.
  auto output_y3_flag = with_act_calc_;

  auto alltoall_allgather_batch_matmul_cnode =
    PreProcessAndCreateAllToAllAllGatherBatchMatMulCNode(func_graph, output_y2_flag, output_y3_flag);

  MS_CHECK_TRUE_RET(alltoall_allgather_batch_matmul_cnode != nullptr, nullptr);

  // 3. Replace graph
  ReplaceGraph(func_graph, alltoall_allgather_batch_matmul_cnode, reshape_before_batch_matmul_user_other_users,
               bias_add_other_users, output_y2_flag, output_y3_flag);

  MS_LOG(DEBUG) << "Process succeed. The new fusion node is "
                << alltoall_allgather_batch_matmul_cnode->fullname_with_scope();
  return alltoall_allgather_batch_matmul_cnode;
}

bool AllToAllAllGatherBatchMatMulFusion::Run(const FuncGraphPtr &func_graph) {
  if (!IsKbkAclnnMode() || !IsAscendVerison93()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(func_graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  auto mc2_fusion_level = ms_context->get_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL);
  if (mc2_fusion_level == kMC2NotFusion) {
    MS_LOG(DEBUG) << "MC2 fusion level is 0, not enabling AllToAllAllGatherBatchMatMulFusion.";
    return false;
  }

  if (mc2_fusion_level == kMC2FusionForward || mc2_fusion_level == kMC2FusionBackward ||
      mc2_fusion_level == kMC2FusionFull) {
    if (common::IsExecuteSimulation()) {
      MS_LOG(EXCEPTION) << "Not support compute_communication_fusion_level when MS_SIMULATION_LEVEL=3.";
    }
    return NodePass::Run(func_graph);
  } else {
    MS_LOG(DEBUG) << "MC2 fusion level is " << mc2_fusion_level
                  << ", not supporting AllToAllAllGatherBatchMatMulFusion.";
    return false;
  }
}
}  // namespace mindspore::opt
