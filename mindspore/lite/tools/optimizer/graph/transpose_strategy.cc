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
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "ops/crop.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/op_utils.h"
#include "tools/anf_exporter/fetch_content.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kFirstInput = 1;
constexpr size_t kHalfDivisor = 2;
constexpr size_t kOnnxStridedSlice = 6;
constexpr int kPaddingListLength = 8;
STATUS GetPostNodes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, std::vector<AnfNodePtr> *out_nodes) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && out_nodes != nullptr);
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

bool JudgeIs4DInput(NodeInferShape *node_infer_shape, const CNodePtr &cnode) {
  MS_ASSERT(node_infer_shape != nullptr && cnode != nullptr);
  auto shape = node_infer_shape->GetInputShape(cnode, 1);
  if (shape.size() != kInputSizeFour) {
    if (cnode->size() > kInputSizeTwo) {
      shape = node_infer_shape->GetInputShape(cnode, kInputIndexTwo);
      if (shape.size() != kInputSizeFour && !shape.empty()) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

std::vector<int> TransformOpAxesAttr(const std::vector<int> &origin_axes, FormatTransNodeType trans_type) {
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

int TransformAttrByAxes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index,
                        const std::vector<int> &axes, FormatTransNodeType trans_type,
                        NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
  if (input_index >= cnode->size() || axes.empty()) {
    return lite::RET_ERROR;
  }
  auto origin_input = node_infer_shape->GetIntVecInput(cnode, input_index);
  if (origin_input.size() != axes.size()) {
    return lite::RET_ERROR;
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
  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "BuildIntVecParameterNode failed");
  func_graph->manager()->Replace(cnode->input(input_index), param_node);
  return lite::RET_OK;
}

STATUS ChangeCommonOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type,
                      NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
  if (prim->GetAttr(ops::kAxis) == nullptr) {
    return lite::RET_NOT_SUPPORT;
  }
  MS_CHECK_TRUE_MSG(prim->GetAttr(ops::kAxis) != nullptr, lite::RET_NULL_PTR, "GetAttr Failed.");
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

STATUS ChangeOpCrop(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type,
                    NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  auto crop_prim = GetValueNode<std::shared_ptr<ops::Crop>>(cnode->input(0));
  if (crop_prim == nullptr) {
    MS_LOG(ERROR) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(crop_prim->GetAttr(ops::kAxis) != nullptr, lite::RET_ERROR);
  auto axis = crop_prim->get_axis();
  if (axis < 0) {
    axis += kInputSizeFour;
  }
  MS_ASSERT(axis >= 0 && axis < kInputSizeFour);
  MS_CHECK_TRUE_RET(crop_prim->GetAttr(ops::kOffsets) != nullptr, lite::RET_ERROR);
  auto offsets = crop_prim->get_offsets();
  if (trans_type == kNCHW2NHWC) {
    auto new_axis = kNH2NC[axis];
    if (new_axis == 0) {
      MS_CHECK_GE(offsets.size(), kInputIndexFour, lite::RET_ERROR);
      offsets = {offsets[0], offsets[kInputIndexTwo], offsets[kInputIndexThree], offsets[1]};
    } else if (new_axis == kInputIndexThree) {
      MS_CHECK_GE(offsets.size(), kInputIndexThree, lite::RET_ERROR);
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

STATUS ChangeOpPad(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type,
                   NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  if (cnode->size() < kInputSizeThree) {
    MS_LOG(ERROR) << "pad op need three inputs.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto second_input = cnode->input(kInputIndexTwo);
  lite::DataInfo data_info;
  int status;
  if (utils::isa<Parameter>(second_input)) {
    status = lite::FetchDataFromParameterNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &data_info);
  } else if (utils::isa<ValueNode>(second_input)) {
    status = lite::FetchDataFromValueNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &data_info);
  } else {
    return lite::RET_NOT_SUPPORT;
  }
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get paddings failed.";
    return status;
  }
  if (std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1, std::multiplies<int>()) !=
      kPaddingListLength) {
    return lite::RET_OK;
  }
  std::vector<std::vector<int32_t>> padding_list(kInputSizeFour, std::vector<int32_t>(kInputSizeTwo));
  auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
  for (int i = 0; i < kPaddingListLength; ++i) {
    padding_list[i / kInputIndexTwo][i % kInputIndexTwo] = *data;
    data += 1;
  }
  if (trans_type == kNCHW2NHWC) {
    auto chanel_pad = padding_list[1];
    padding_list.erase(padding_list.begin() + 1);
    padding_list.push_back(chanel_pad);
  } else {
    auto chanel_pad = padding_list.back();
    padding_list.pop_back();
    padding_list.insert(padding_list.begin() + 1, chanel_pad);
  }
  auto param_node =
    BuildIntVec2DParameterNode(func_graph, padding_list, cnode->input(kInputIndexTwo)->fullname_with_scope());
  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_NULL_PTR, "BuildParameterNode Failed");
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->Replace(cnode->input(kInputIndexTwo), param_node);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
  if (prim->GetAttr(ops::kPaddings) != nullptr) {
    std::vector<std::vector<int64_t>> padding_attr;
    (void)std::transform(padding_list.begin(), padding_list.end(), std::back_inserter(padding_attr),
                         [](const std::vector<int> &val) { return std::vector<int64_t>(val.begin(), val.end()); });
    prim->AddAttr(ops::kPaddings, MakeValue(padding_attr));
  }
  return lite::RET_OK;
}

STATUS ChangeOpSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type,
                     NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
  if (trans_type == kNONE) {
    MS_LOG(ERROR) << "trans_type is invalid.";
    return lite::RET_ERROR;
  }
  for (size_t i = 2; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      return lite::RET_NOT_SUPPORT;
    }
  }
  auto shape = node_infer_shape->GetInputShape(cnode, kInputIndexTwo);
  if (shape.empty()) {
    return lite::RET_NOT_SUPPORT;
  }
  int element_num = shape.front();
  auto prim = GetValueNode<std::shared_ptr<ops::SliceFusion>>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "GetValueNode failed");
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
    if (TransformAttrByAxes(func_graph, cnode, i, axes, trans_type, node_infer_shape) != RET_OK) {
      MS_LOG(ERROR) << "Transform axes failed.";
      return RET_ERROR;
    }
  }
  auto tmp_axes = TransformOpAxesAttr(axes, trans_type);
  std::vector<int64_t> new_axes(tmp_axes.begin(), tmp_axes.end());
  prim->set_axes(new_axes);
  return lite::RET_OK;
}

STATUS ChangeOpStrideSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type,
                           NodeInferShape *node_infer_shape) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && node_infer_shape != nullptr);
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
  std::vector<int> axes = node_infer_shape->GetIntVecInput(cnode, kInputIndexFour);
  if (axes.empty()) {
    MS_LOG(ERROR) << "strided slice input invalid.";
    return lite::RET_ERROR;
  }
  for (size_t index = 2; index < cnode->size(); ++index) {
    if (index == kInputIndexFour) {
      continue;
    }
    if (TransformAttrByAxes(func_graph, cnode, index, axes, trans_type, node_infer_shape) != RET_OK) {
      MS_LOG(ERROR) << "transform axes failed.";
      return lite::RET_ERROR;
    }
  }
  auto cur_axes = TransformOpAxesAttr(axes, trans_type);
  auto param_node =
    BuildIntVecParameterNode(func_graph, cur_axes, cnode->input(kInputIndexFour)->fullname_with_scope());
  MS_CHECK_TRUE_MSG(param_node != nullptr, RET_ERROR, "BuildIntVecParameterNode failed");
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->Replace(cnode->input(kInputIndexFour), param_node);
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
  auto graph_inputs = func_graph->get_inputs();
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i)) ||
        std::find(graph_inputs.begin(), graph_inputs.end(), cnode->input(i)) != graph_inputs.end()) {
      in_nodes.push_back(cnode->input(i));
    }
  }
  if (!IsInOutCanFuison(in_nodes, &trans_count, &trans_info->pre_)) {
    return false;
  }
  std::vector<AnfNodePtr> out_nodes;
  if (GetPostNodes(func_graph, cnode, &out_nodes) != lite::RET_OK) {
    return false;
  }
  if (!IsInOutCanFuison(out_nodes, &trans_count, &trans_info->post_)) {
    return false;
  }
  if (trans_info->pre_ == trans_info->post_) {
    return false;
  }
  auto total_node_count = in_nodes.size() + out_nodes.size();
  bool can_insert = trans_count > total_node_count / kHalfDivisor;
  if (CheckPrimitiveType(cnode, prim::kPrimActivation)) {
    auto prim_act = GetValueNode<std::shared_ptr<ops::Activation>>(cnode->input(0));
    MS_CHECK_TRUE_MSG(prim_act != nullptr, false, "GetValueNode Failed");
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

bool TransposeStrategy::CanChangeOpAxis(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "GetValueNode Failed");
  if (!IsDynamicFormatOp(prim->name())) {
    return false;
  }
  if (IsDynamicFormatOpWithAxis(prim->name()) && !JudgeIs4DInput(&node_infer_shape_, cnode)) {
    return false;
  }
  if (CheckPrimitiveType(cnode, prim::kPrimSliceFusion) || CheckPrimitiveType(cnode, prim::kPrimStridedSlice) ||
      CheckPrimitiveType(cnode, prim::kPrimPadFusion)) {
    for (size_t i = 2; i < cnode->size(); ++i) {
      if (utils::isa<CNodePtr>(cnode->input(i))) {
        return false;
      }
      if (utils::isa<Parameter>(cnode->input(i)) && !cnode->input(i)->cast<ParameterPtr>()->has_default()) {
        return false;
      }
    }
    if (CheckPrimitiveType(cnode, prim::kPrimStridedSlice) && cnode->size() != kOnnxStridedSlice) {
      return false;
    }
  } else if (IsDynamicFormatOpWithAxis(prim->name())) {
    if (prim->GetAttr(ops::kAxis) == nullptr) {
      return false;
    }
  }
  return true;
}

STATUS TransposeStrategy::ChangeOpAxis(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       FormatTransNodeType trans_type) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
  if (IsDynamicFormatOpWithAxis(prim->name()) && !JudgeIs4DInput(&node_infer_shape_, cnode)) {
    return lite::RET_NOT_SUPPORT;
  }
  std::map<std::string,
           std::function<STATUS(const FuncGraphPtr &, const CNodePtr &, FormatTransNodeType, NodeInferShape *)>>
    process_funcs = {
      {prim::kPrimConcat->name(), ChangeCommonOp},     {prim::kPrimSplit->name(), ChangeCommonOp},
      {prim::kPrimCrop->name(), ChangeOpCrop},         {prim::kPrimPadFusion->name(), ChangeOpPad},
      {prim::kPrimSliceFusion->name(), ChangeOpSlice}, {prim::kPrimStridedSlice->name(), ChangeOpStrideSlice}};
  auto iter = process_funcs.find(prim->name());
  if (iter != process_funcs.end()) {
    return iter->second(func_graph, cnode, trans_type, &node_infer_shape_);
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
  MS_ASSERT(base_node != nullptr);
  size_t input_index = before ? index : node_users.front().second;
  auto shape = node_infer_shape_.GetInputShape(base_node, input_index);
  if (!shape.empty() && shape.size() != kNH2NC.size()) {
    return lite::RET_NO_CHANGE;
  }
  return lite::RET_OK;
}

bool TransposeStrategy::IsInOutCanFuison(const std::vector<AnfNodePtr> &nodes, size_t *trans_count,
                                         FormatTransNodeType *trans_type) {
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

void TransposeStrategy::DecidePreAndPostTransType(TransTypePair *trans_info, TransTypePair *trans_insert_info) const {
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
}  // namespace opt
}  // namespace mindspore
