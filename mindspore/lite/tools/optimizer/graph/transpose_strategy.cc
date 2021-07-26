/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/graph/transpose_strategy.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "ops/crop.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/op_utils.h"
#include "ops/strided_slice.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kFirstInput = 1;
constexpr size_t kHalfDivisor = 2;
constexpr size_t kOnnxStridedSlice = 6;
STATUS GetPostNodes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, std::vector<AnfNodePtr> *out_nodes) {
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_ERROR;
  }
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(ERROR) << "cnode is isolated.";
    return lite::RET_ERROR;
  }
  std::transform(node_users.begin(), node_users.end(), std::back_inserter(*out_nodes),
                 [](const std::pair<AnfNodePtr, int> &node_user) { return node_user.first; });
  return lite::RET_OK;
}
}  // namespace

AnfNodePtr TransposeStrategy::TransposePairFuseWhenInsert(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                          const std::vector<int> &perm, bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr trans_input_node = before ? cnode->input(index) : cnode;
  // judge pair transpose after insert.
  if (CheckPrimitiveType(trans_input_node, prim::kPrimTranspose)) {
    std::vector<int> trans_perm;
    auto input_cnode = trans_input_node->cast<CNodePtr>();
    if (input_cnode == nullptr) {
      MS_LOG(ERROR) << "input node is invalid.";
      return nullptr;
    }
    if (GetTransposePerm(input_cnode, &trans_perm) != lite::RET_OK) {
      MS_LOG(ERROR) << "transpose perm get failed.";
      return nullptr;
    }
    if ((perm == kNH2NC && trans_perm == kNC2NH) || (perm == kNC2NH && trans_perm == kNH2NC)) {
      return input_cnode->input(kFirstInput);
    }
  }
  // insert depend on shape
  return TransposeDependOnShape(func_graph, cnode, perm, before, index);
}

AnfNodePtr TransposeStrategy::TransposeDependOnShape(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                     const std::vector<int> &perm, bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr trans_input_node = before ? cnode->input(index) : cnode;
  auto status = TransposeInsertDependOnShape(func_graph, cnode, before, index);
  if (status == lite::RET_ERROR) {
    return nullptr;
  } else if (status == lite::RET_NO_CHANGE) {
    return before ? cnode->input(index) : cnode;
  }
  // insert tranpsoe
  std::string trans_name =
    before ? cnode->fullname_with_scope() + "_pre" + std::to_string(index - 1) : cnode->fullname_with_scope() + "_post";
  auto trans_insert_node = GenTransposeNode(func_graph, trans_input_node, perm, trans_name);
  return trans_insert_node;
}

bool TransposeStrategy::CanFusionIfInsert(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                          TransTypePair *trans_info, TransTypePair *trans_insert_info) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  MS_ASSERT(pre_type != nullptr && post_type != nullptr);
  size_t trans_count = 0;
  std::vector<AnfNodePtr> in_nodes;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      in_nodes.push_back(cnode->input(i));
    }
  }
  if (!IsInOutCanFuison(func_graph, in_nodes, &trans_count, &trans_info->pre_)) {
    return false;
  }
  std::vector<AnfNodePtr> out_nodes;
  if (GetPostNodes(func_graph, cnode, &out_nodes) != lite::RET_OK) {
    return false;
  }
  if (!IsInOutCanFuison(func_graph, out_nodes, &trans_count, &trans_info->post_)) {
    return false;
  }
  if (trans_info->pre_ == trans_info->post_) {
    return false;
  }
  auto total_node_count = in_nodes.size() + out_nodes.size();
  bool can_insert = trans_count > total_node_count / kHalfDivisor;
  if (CheckPrimitiveType(cnode, prim::kPrimActivation)) {
    auto prim_act = GetValueNode<std::shared_ptr<ops::Activation>>(cnode->input(0));
    MS_ASSERT(prim_act != nullptr);
    if (prim_act->get_activation_type() == mindspore::ActivationType::LEAKY_RELU) {
      can_insert = trans_count >= total_node_count / kHalfDivisor;
    }
  }
  if (CheckPrimitiveType(cnode, prim::kPrimSplit) || CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    can_insert = trans_count >= total_node_count / kHalfDivisor;
  }
  if (!can_insert) {
    return can_insert;
  }
  DecidePreAndPostTransType(trans_info, trans_insert_info);
  return can_insert;
}

bool TransposeStrategy::CanChangeOpAxis(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto shape = node_infer_shape_.GetInputShape(cnode, 1);
  if (shape.size() != kInputSizeFour) {
    if (cnode->size() > kInputSizeTwo) {
      shape = node_infer_shape_.GetInputShape(cnode, kInputIndexTwo);
      if (shape.size() != kInputSizeFour && !shape.empty()) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (CheckPrimitiveType(cnode, prim::kPrimConcat) || CheckPrimitiveType(cnode, prim::kPrimSplit)) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->GetAttr(ops::kAxis) == nullptr) {
      return false;
    }
  }
  if (CheckPrimitiveType(cnode, prim::kPrimSliceFusion) || CheckPrimitiveType(cnode, prim::kPrimStridedSlice)) {
    for (size_t i = 2; i < cnode->size(); ++i) {
      if (utils::isa<CNodePtr>(cnode->input(i))) {
        return false;
      }
    }
    if (CheckPrimitiveType(cnode, prim::kPrimStridedSlice) && cnode->size() != kOnnxStridedSlice) {
      return false;
    }
  }
  return true;
}

STATUS TransposeStrategy::ChangeOpAxis(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       FormatTransNodeType trans_type) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto shape = node_infer_shape_.GetInputShape(cnode, 1);
  if (shape.size() != kInputSizeFour) {
    if (cnode->size() > kInputSizeTwo) {
      shape = node_infer_shape_.GetInputShape(cnode, kInputIndexTwo);
      if (shape.size() != kInputSizeFour && !shape.empty()) {
        return lite::RET_NOT_SUPPORT;
      }
    } else {
      return lite::RET_NOT_SUPPORT;
    }
  }
  if (CheckPrimitiveType(cnode, prim::kPrimConcat) || CheckPrimitiveType(cnode, prim::kPrimSplit)) {
    return ChangeCommonOp(cnode, trans_type);
  }
  if (CheckPrimitiveType(cnode, prim::kPrimCrop)) {
    return ChangeOpCrop(cnode, trans_type);
  }
  if (CheckPrimitiveType(cnode, prim::kPrimSliceFusion)) {
    return ChangeOpSlice(func_graph, cnode, trans_type);
  }
  if (CheckPrimitiveType(cnode, prim::kPrimStridedSlice)) {
    return ChangeOpStrideSlice(func_graph, cnode, trans_type);
  }
  return lite::RET_OK;
}

STATUS TransposeStrategy::TransposeInsertDependOnShape(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                       bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_ERROR;
  }
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(ERROR) << "cnode is isolated.";
    return lite::RET_ERROR;
  }
  if (!utils::isa<CNodePtr>(node_users.front().first)) {
    return lite::RET_ERROR;
  }
  CNodePtr base_node = before ? cnode : node_users.front().first->cast<CNodePtr>();
  size_t input_index = before ? index : node_users.front().second;
  auto shape = node_infer_shape_.GetInputShape(base_node, input_index);
  if (!shape.empty() && shape.size() != kNH2NC.size()) {
    return lite::RET_NO_CHANGE;
  }
  return lite::RET_OK;
}

bool TransposeStrategy::IsInOutCanFuison(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                                         size_t *trans_count, FormatTransNodeType *trans_type) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(trans_count != nullptr && trans_type != nullptr);
  for (auto &node : nodes) {
    if (CheckPrimitiveType(node, prim::kPrimTranspose)) {
      FormatTransNodeType cur_type;
      std::vector<int> perm;
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        return false;
      }
      if (GetTransposePerm(cnode, &perm) != lite::RET_OK) {
        return false;
      }
      if (perm == kNH2NC) {
        cur_type = kNHWC2NCHW;
      } else if (perm == kNC2NH) {
        cur_type = kNCHW2NHWC;
      } else {
        return false;
      }
      if (*trans_type == kNONE) {
        *trans_type = cur_type;
      } else if (*trans_type != cur_type) {
        return false;
      }
      *trans_count += 1;
    }
  }
  return true;
}

void TransposeStrategy::DecidePreAndPostTransType(TransTypePair *trans_info, TransTypePair *trans_insert_info) {
  if (trans_info->pre_ == trans_info->post_) {
    return;
  }
  if (trans_info->pre_ != kNONE && trans_info->post_ != kNONE) {
    trans_insert_info->pre_ = trans_info->pre_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
    trans_insert_info->post_ = trans_info->post_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
  } else if (trans_info->pre_ == kNONE) {
    trans_insert_info->pre_ = trans_info->post_ == kNHWC2NCHW ? kNHWC2NCHW : kNCHW2NHWC;
    trans_insert_info->post_ = trans_info->post_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
  } else {
    trans_insert_info->pre_ = trans_info->pre_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
    trans_insert_info->post_ = trans_info->pre_ == kNHWC2NCHW ? kNHWC2NCHW : kNCHW2NHWC;
  }
}

STATUS TransposeStrategy::ChangeCommonOp(const CNodePtr &cnode, FormatTransNodeType trans_type) {
  MS_ASSERT(cnode != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ASSERT(prim != nullptr);
  if (prim->GetAttr(ops::kAxis) == nullptr) {
    return lite::RET_NOT_SUPPORT;
  }
  auto axis = GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  if (axis < 0) {
    axis += kInputSizeFour;
  }
  auto new_axis = kNH2NC[axis];
  if (trans_type == kNHWC2NCHW) {
    new_axis = kNC2NH[axis];
  }
  prim->AddAttr(ops::kAxis, MakeValue<int64_t>(new_axis));
  return lite::RET_OK;
}

STATUS TransposeStrategy::ChangeOpCrop(const CNodePtr &cnode, FormatTransNodeType trans_type) {
  MS_ASSERT(cnode != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  auto crop_prim = GetValueNode<std::shared_ptr<ops::Crop>>(cnode->input(0));
  if (crop_prim == nullptr) {
    MS_LOG(ERROR) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  auto axis = crop_prim->get_axis();
  if (axis < 0) {
    axis += kInputSizeFour;
  }
  MS_ASSERT(axis >= 0 && axis < kInputSizeFour);
  auto offsets = crop_prim->get_offsets();
  if (trans_type == kNCHW2NHWC) {
    auto new_axis = kNH2NC[axis];
    if (new_axis == 0) {
      offsets = {offsets[0], offsets[kInputIndexTwo], offsets[kInputIndexThree], offsets[1]};
    } else if (new_axis == kInputIndexThree) {
      offsets = {offsets[1], offsets[kInputIndexTwo], offsets[0]};
    } else {
      offsets.push_back(0);
    }
    crop_prim->set_axis(new_axis);
    crop_prim->set_offsets(offsets);
  } else {
    auto new_axis = kNC2NH[axis];
    if (new_axis == 0) {
      offsets = {offsets[0], offsets[kInputIndexThree], offsets[1], offsets[kInputIndexTwo]};
    } else if (new_axis == kInputIndexThree) {
      offsets = {offsets[kInputIndexTwo], offsets[0], offsets[1]};
    } else {
      offsets.pop_back();
    }
    crop_prim->set_axis(new_axis);
    crop_prim->set_offsets(offsets);
  }
  return lite::RET_OK;
}

STATUS TransposeStrategy::ChangeOpSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                        FormatTransNodeType trans_type) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  for (size_t i = 2; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      return lite::RET_NOT_SUPPORT;
    }
  }
  auto shape = node_infer_shape_.GetInputShape(cnode, kInputIndexTwo);
  if (shape.empty()) {
    return lite::RET_NOT_SUPPORT;
  }
  int element_num = shape.front();
  auto prim = GetValueNode<std::shared_ptr<ops::SliceFusion>>(cnode->input(0));
  std::vector<int> axes;
  if (prim->GetAttr(ops::kAxes) == nullptr || prim->get_axes().empty()) {
    for (int index = 0; index < element_num; ++index) {
      axes.push_back(index);
    }
  } else {
    auto origin_axes = prim->get_axes();
    std::transform(origin_axes.begin(), origin_axes.end(), std::back_inserter(axes),
                   [](int64_t v) { return static_cast<int>(v); });
  }
  for (size_t i = 2; i < cnode->size(); ++i) {
    TransformAttrByAxes(func_graph, cnode, i, axes, trans_type);
  }
  auto tmp_axes = TransformOpAxesAttr(axes, trans_type);
  std::vector<int64_t> new_axes(tmp_axes.begin(), tmp_axes.end());
  prim->set_axes(new_axes);
  return lite::RET_OK;
}

STATUS TransposeStrategy::ChangeOpStrideSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                              FormatTransNodeType trans_type) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kOnnxStridedSlice) {
    return lite::RET_NOT_SUPPORT;
  }
  for (size_t i = 2; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      return lite::RET_NOT_SUPPORT;
    }
  }
  std::vector<int> axes = node_infer_shape_.GetIntVecInput(cnode, kInputIndexFour);
  if (axes.empty()) {
    MS_LOG(ERROR) << "strided slice input invalid.";
    return lite::RET_ERROR;
  }
  for (size_t index = 2; index < cnode->size(); ++index) {
    if (index == kInputIndexFour) {
      continue;
    }
    TransformAttrByAxes(func_graph, cnode, index, axes, trans_type);
  }
  auto cur_axes = TransformOpAxesAttr(axes, trans_type);
  auto param_node =
    BuildIntVecParameterNode(func_graph, cur_axes, cnode->input(kInputIndexFour)->fullname_with_scope());
  func_graph->manager()->Replace(cnode->input(kInputIndexFour), param_node);
  return lite::RET_OK;
}

void TransposeStrategy::TransformAttrByAxes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index,
                                            const std::vector<int> &axes, FormatTransNodeType trans_type) {
  if (cnode == nullptr || input_index >= cnode->size() || axes.empty()) {
    return;
  }
  auto origin_input = node_infer_shape_.GetIntVecInput(cnode, input_index);
  if (origin_input.size() != axes.size()) {
    return;
  }
  std::vector<int> cur_input;
  for (int dim = 0; dim < static_cast<int>(kInputSizeFour); ++dim) {
    for (size_t index = 0; index < axes.size(); ++index) {
      int axis = axes[index];
      if (axis < 0) {
        axis += kInputSizeFour;
      }
      MS_ASSERT(axis >= 0 && axis < kInputSizeFour);
      int cur_axis = kNH2NC[axis];
      if (trans_type == kNHWC2NCHW) {
        cur_axis = kNC2NH[axis];
      }
      if (cur_axis == dim) {
        cur_input.push_back(origin_input[index]);
      }
    }
  }
  auto param_node = BuildIntVecParameterNode(func_graph, cur_input, cnode->input(input_index)->fullname_with_scope());
  func_graph->manager()->Replace(cnode->input(input_index), param_node);
}

std::vector<int> TransposeStrategy::TransformOpAxesAttr(const std::vector<int> &origin_axes,
                                                        FormatTransNodeType trans_type) {
  std::vector<int> cur_axes;
  for (size_t i = 0; i < origin_axes.size(); ++i) {
    int axis = origin_axes[i];
    if (axis < 0) {
      axis += kInputSizeFour;
    }
    MS_ASSERT(axis >= 0 && axis < kInputSizeFour);
    int cur_axis = kNH2NC[axis];
    if (trans_type == kNHWC2NCHW) {
      cur_axis = kNC2NH[axis];
    }
    cur_axes.push_back(cur_axis);
  }
  std::sort(cur_axes.begin(), cur_axes.end());
  return cur_axes;
}
}  // namespace opt
}  // namespace mindspore
