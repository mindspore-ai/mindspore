/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/mul_reduce_fusion.h"
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/squeeze.h"
#include "ops/op_name.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kReciprocalFirstIndex = -1;
constexpr int kReciprocalSecondIndex = -2;
}  // namespace

bool MulReduceFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  auto ret = preprocessor_.Run(func_graph);
  if (ret == lite::RET_NOT_SUPPORT) {
    return true;
  }
  if (ret != lite::RET_OK) {
    return false;
  }
  auto &shape_container = preprocessor_.GetShapeContainer();
  std::vector<CNodePtr> reduce_ops;
  for (auto infos : shape_container) {
    if (!utils::isa<CNode>(infos.first)) {
      continue;
    }
    if (!CheckPrimitiveType(infos.first, prim::kPrimReduceFusion)) {
      continue;
    }
    reduce_ops.push_back(infos.first->cast<CNodePtr>());
  }
  for (auto reduce_op : reduce_ops) {
    ret = ProcessOp(func_graph, reduce_op);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "mul-reduce fusion process failed.";
      return false;
    }
  }
  ret = PostProcess(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "mul-reduce fusion post-process failed.";
    return false;
  }
  return true;
}

int MulReduceFusion::ProcessOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto is_meet_cond = CheckBasicCond(func_graph, cnode);
  if (!is_meet_cond) {
    return lite::RET_OK;
  }
  bool need_post_mul = false;
  if (reduce_mode_ == ReduceMode::Reduce_Mean) {
    auto ret = ProcessGather(func_graph);
    if (ret == lite::RET_NOT_SUPPORT) {
      need_post_mul = true;
    } else if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Process Gather op failed.";
      return lite::RET_ERROR;
    }
  }
  if (!keep_dim_) {
    auto ret = GenerateSqueeze(func_graph, cnode);
    if (ret != lite::RET_OK) {
      return lite::RET_ERROR;
    }
  }
  if (need_post_mul) {
    auto ret = GenerateMul(func_graph, cnode);
    if (ret != lite::RET_OK) {
      return lite::RET_ERROR;
    }
  }
  auto ret = GenerateMatmul(func_graph, cnode);
  if (ret != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int MulReduceFusion::ProcessGather(const FuncGraphPtr &func_graph) {
  MS_ASSERT(gather_.size() > C1NUM);
  auto gather_table = gather_->input(1);
  if (gather_table == nullptr || utils::isa<CNode>(gather_table)) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(gather_, 1, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Fetch const data of gather failed.");
  if (data_info.data_type_ != kNumberTypeFloat && data_info.data_type_ != kNumberTypeFloat32) {
    return lite::RET_NOT_SUPPORT;
  }
  if (data_info.data_ptr_ == nullptr) {
    return lite::RET_NOT_SUPPORT;
  }
  auto *float_data = static_cast<float *>(data_info.data_ptr_);
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  for (int64_t i = 0; i < element_num; ++i) {
    float_data[i] *= coeff_;
  }
  return lite::RET_OK;
}

int MulReduceFusion::PostProcess(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  if (squeeze_infos_.empty()) {
    return lite::RET_OK;
  }
  std::set<CNodePtr> concat_ops;
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto &node_users = manager->node_users();
  for (auto &squeeze : squeeze_infos_) {
    auto &node_user = node_users[squeeze.first];
    for (auto &user : node_user) {
      auto node = user.first;
      if (!utils::isa<CNode>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (CheckPrimitiveType(cnode, prim::kPrimConcat)) {
        concat_ops.insert(cnode);
      }
    }
  }
  for (auto &concat : concat_ops) {
    auto ret = PostProcessSqueezeWithConcat(func_graph, concat);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "mul-reduce-fusion's PostProcess failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int MulReduceFusion::PostProcessSqueezeWithConcat(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (!CheckConcatOp(func_graph, cnode)) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  for (size_t i = 1; i < cnode->size(); ++i) {
    manager->SetEdge(cnode, i, cnode->input(i)->cast<CNodePtr>()->input(1));
  }
  auto concat_prim = GetCNodePrimitive(cnode);
  MS_ASSERT(concat_prim != nullptr);
  concat_prim->AddAttr(ops::kAxis, MakeValue<int64_t>(concat_axis_));
  auto &node_users = manager->node_users();
  auto concat_users = node_users[cnode];
  CNodePtr post_squeeze{nullptr};
  for (auto &user : concat_users) {
    if (CheckPrimitiveType(user.first, prim::kPrimReshape)) {
      continue;
    }
    if (post_squeeze == nullptr) {
      auto squeeze = std::make_shared<ops::Squeeze>();
      MS_CHECK_TRUE_MSG(squeeze != nullptr, lite::RET_ERROR, "Squeeze create failed.");
      squeeze->set_axis(std::vector<int64_t>{axis_});
      auto squeeze_prim = squeeze->GetPrim();
      MS_CHECK_TRUE_MSG(squeeze_prim != nullptr, lite::RET_ERROR, "Squeeze create failed.");
      post_squeeze = func_graph->NewCNode(squeeze_prim, {cnode});
      MS_CHECK_TRUE_MSG(post_squeeze != nullptr, lite::RET_ERROR, "Squeeze-cnode create failed.");
      post_squeeze->set_fullname_with_scope(cnode->fullname_with_scope() + "/Squeeze");
    }
    manager->SetEdge(user.first, user.second, post_squeeze);
  }
  return lite::RET_OK;
}

int MulReduceFusion::GenerateMatmul(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is a nullptr.");
  auto mul_op = cnode->input(1)->cast<CNodePtr>();  // which has been checked before.
  if (exchange_) {
    manager->SetEdge(cnode, 1, mul_op->input(kInputIndexTwo));
    manager->SetEdge(cnode, kInputIndexTwo, mul_op->input(1));
  } else {
    manager->SetEdge(cnode, 1, mul_op->input(1));
    manager->SetEdge(cnode, kInputIndexTwo, mul_op->input(kInputIndexTwo));
  }
  auto matmul_prim = std::make_shared<ops::MatMulFusion>();
  MS_CHECK_TRUE_MSG(matmul_prim != nullptr, lite::RET_ERROR, "Matmul create failed.");
  auto matmul_prim_c = matmul_prim->GetPrim();
  MS_CHECK_TRUE_MSG(matmul_prim_c != nullptr, lite::RET_ERROR, "Matmul create failed.");
  matmul_prim->set_transpose_a(transpose_a_);
  matmul_prim->set_transpose_b(transpose_b_);
  MS_ASSERT(cnode->input(0) != nullptr);
  auto reduce_prim_carrier = cnode->input(0)->cast<ValueNodePtr>();
  MS_ASSERT(reduce_prim_carrier != nullptr);
  reduce_prim_carrier->set_value(matmul_prim_c);
  return lite::RET_OK;
}

int MulReduceFusion::GenerateSqueeze(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is a nullptr.");
  auto squeeze = std::make_shared<ops::Squeeze>();
  MS_CHECK_TRUE_MSG(squeeze != nullptr, lite::RET_ERROR, "Squeeze create failed.");
  squeeze->set_axis(std::vector<int64_t>{axis_});
  auto squeeze_prim = squeeze->GetPrim();
  MS_CHECK_TRUE_MSG(squeeze_prim != nullptr, lite::RET_ERROR, "Squeeze create failed.");
  auto squeeze_cnode = func_graph->NewCNode(squeeze_prim, {cnode});
  MS_CHECK_TRUE_MSG(squeeze_cnode != nullptr, lite::RET_ERROR, "Squeeze-cnode create failed.");
  auto mul_op = cnode->input(1);
  MS_ASSERT(mul_op != nullptr);
  squeeze_cnode->set_fullname_with_scope(mul_op->fullname_with_scope() + "/Squeeze");
  auto success = manager->Replace(cnode, squeeze_cnode);
  MS_CHECK_TRUE_MSG(success, lite::RET_ERROR, "Replace old node failed.");
  auto &shape_infos = preprocessor_.GetShapeContainer();
  MS_ASSERT(shape_infos.find(mul_op) != shape_infos.end());
  auto &out_shape_infos = shape_infos.at(mul_op).second;
  MS_ASSERT(!out_shape_infos.empty());
  squeeze_infos_[squeeze_cnode] = std::make_pair(axis_, out_shape_infos.front().size() - 1);
  return lite::RET_OK;
}

int MulReduceFusion::GenerateMul(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (coeff_ == 1.0f) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is a nullptr.");
  auto mul = std::make_shared<ops::MulFusion>();
  MS_CHECK_TRUE_MSG(mul != nullptr, lite::RET_ERROR, "Mul create failed.");
  auto mul_prim = mul->GetPrim();
  MS_CHECK_TRUE_MSG(mul_prim != nullptr, lite::RET_ERROR, "Mul create failed.");
  auto old_mul_op = cnode->input(1);
  MS_ASSERT(old_mul_op != nullptr);
  auto second_input_node =
    BuildFloatValueParameterNode(func_graph, coeff_, old_mul_op->fullname_with_scope() + "/scale");
  MS_CHECK_TRUE_MSG(second_input_node != nullptr, lite::RET_ERROR, "Mul second-input create failed.");
  auto mul_cnode = func_graph->NewCNode(mul_prim, {cnode, second_input_node});
  MS_CHECK_TRUE_MSG(mul_cnode != nullptr, lite::RET_ERROR, "Mul-cnode create failed.");
  mul_cnode->set_fullname_with_scope(old_mul_op->fullname_with_scope());
  auto success = manager->Replace(cnode, mul_cnode);
  MS_CHECK_TRUE_MSG(success, lite::RET_ERROR, "Replace old node failed.");
  return lite::RET_OK;
}

bool MulReduceFusion::CheckBasicCond(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree) {
    return false;
  }
  if (IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_ASSERT(prim != nullptr);
  bool is_to_end = prim->GetAttr(ops::kReduceToEnd) != nullptr && GetValue<bool>(prim->GetAttr(ops::kReduceToEnd));
  if (is_to_end) {
    return false;
  }
  keep_dim_ = prim->GetAttr(ops::kKeepDims) != nullptr && GetValue<bool>(prim->GetAttr(ops::kKeepDims));
  auto mode_attr = prim->GetAttr(ops::kMode);
  if (mode_attr == nullptr) {
    return false;
  }
  reduce_mode_ = GetValue<int64_t>(mode_attr);
  if (reduce_mode_ != ReduceMode::Reduce_Sum && reduce_mode_ != ReduceMode::Reduce_Mean) {
    return false;
  }
  auto first_input = cnode->input(1);
  if (!utils::isa<CNode>(first_input)) {
    return false;
  }
  if (!CheckPrimitiveType(first_input, prim::kPrimMulFusion)) {
    return false;
  }
  if (IsMarkedTrainOp(first_input->cast<CNodePtr>())) {
    return false;
  }
  auto mul_prim = GetCNodePrimitive(first_input);
  MS_ASSERT(mul_prim != nullptr);
  auto act_type = mul_prim->GetAttr(ops::kActivationType) == nullptr
                    ? ActivationType::NO_ACTIVATION
                    : GetValue<int64_t>(mul_prim->GetAttr(ops::kActivationType));
  if (act_type != ActivationType::NO_ACTIVATION) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, first_input)) {
    return false;
  }
  bool is_axis_meet = CheckAxisCond(cnode);
  if (!is_axis_meet) {
    return false;
  }
  bool is_shape_meet = CheckShapeCond(cnode);
  if (!is_shape_meet) {
    return false;
  }
  return CheckGatherOp(func_graph, cnode);
}

bool MulReduceFusion::CheckAxisCond(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto &shape_container = preprocessor_.GetShapeContainer();
  auto first_input = cnode->input(1);
  if (shape_container.find(first_input) == shape_container.end()) {
    return false;
  }
  if (shape_container.at(first_input).second.empty()) {
    return false;
  }
  auto in_shape = shape_container.at(first_input).second.front();
  auto second_input = cnode->input(kInputIndexTwo);
  if (second_input == nullptr || utils::isa<CNode>(second_input)) {
    return false;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "Fetch reduceOp's axis failed.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  if (data_info.data_ptr_ == nullptr || element_num != 1) {
    return false;
  }
  if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
    axis_ = *(static_cast<int *>(data_info.data_ptr_));
  } else if (data_info.data_type_ == kNumberTypeInt64) {
    axis_ = *(static_cast<int64_t *>(data_info.data_ptr_));
  } else {
    return false;
  }
  if (axis_ > 0) {
    axis_ -= static_cast<int>(in_shape.size());
  }
  if (axis_ != kReciprocalFirstIndex && axis_ != kReciprocalSecondIndex) {
    return false;
  }
  return true;
}

bool MulReduceFusion::CheckShapeCond(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto &shape_container = preprocessor_.GetShapeContainer();
  auto first_input = cnode->input(1);
  if (shape_container.find(first_input) == shape_container.end()) {
    return false;
  }
  if (shape_container.at(first_input).first.size() != kInputSizeTwo) {
    return false;
  }
  auto mul_in0_shape = shape_container.at(first_input).first.front();
  auto mul_in1_shape = shape_container.at(first_input).first.back();
  if (mul_in0_shape.size() < kInputSizeTwo || mul_in1_shape.size() < kInputSizeTwo) {
    return false;
  }
  if (mul_in0_shape.back() <= 0 || mul_in0_shape[mul_in0_shape.size() - C2NUM] <= 0 || mul_in1_shape.back() <= 0 ||
      mul_in1_shape[mul_in1_shape.size() - C2NUM] <= 0) {
    return false;
  }
  if (axis_ == kReciprocalFirstIndex) {
    if (mul_in0_shape.back() != mul_in1_shape.back() ||
        (mul_in0_shape[mul_in0_shape.size() - C2NUM] != 1 && mul_in1_shape[mul_in1_shape.size() - C2NUM] != 1)) {
      return false;
    }
    exchange_ = mul_in1_shape[mul_in1_shape.size() - C2NUM] == 1 ? false : true;
    transpose_a_ = false;
    transpose_b_ = true;
    MS_ASSERT(mul_in0_shape.back() != 0);
    coeff_ = 1.0f / static_cast<float>(mul_in0_shape.back());
    return true;
  }
  if (axis_ == kReciprocalSecondIndex) {
    if (mul_in0_shape[mul_in0_shape.size() - C2NUM] != mul_in1_shape[mul_in1_shape.size() - C2NUM] ||
        (mul_in0_shape.back() != 1 && mul_in1_shape.back() != 1)) {
      return false;
    }
    exchange_ = mul_in0_shape.back() == 1 ? false : true;
    transpose_a_ = true;
    transpose_b_ = false;
    MS_ASSERT(mul_in0_shape[mul_in0_shape.size() - C2NUM] != 0);
    coeff_ = 1.0f / static_cast<float>(mul_in0_shape[mul_in0_shape.size() - C2NUM]);
    return true;
  }
  return false;
}

bool MulReduceFusion::CheckGatherOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (reduce_mode_ == ReduceMode::Reduce_Sum) {
    return true;
  }
  if (reduce_mode_ != ReduceMode::Reduce_Mean) {
    return false;
  }
  auto mul_op = cnode->input(1);
  if (!utils::isa<CNode>(mul_op)) {
    return false;
  }
  auto mul_op_cnode = mul_op->cast<CNodePtr>();
  for (size_t i = 1; i < mul_op_cnode->size(); ++i) {
    if (!utils::isa<CNode>(mul_op_cnode->input(i))) {
      continue;
    }
    if (CheckPrimitiveType(mul_op_cnode->input(i), prim::kPrimGather)) {
      gather_ = mul_op_cnode->input(i)->cast<CNodePtr>();
      break;
    }
  }
  if (gather_ == nullptr) {
    return false;
  }
  if (IsMarkedTrainOp(gather_)) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, gather_)) {
    return lite::RET_OK;
  }
  return true;
}

bool MulReduceFusion::CheckConcatOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  int axis{0};
  int out_dims{0};
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto in_node = cnode->input(i);
    if (!utils::isa<CNode>(in_node)) {
      return false;
    }
    auto in_cnode = in_node->cast<CNodePtr>();
    if (squeeze_infos_.find(in_cnode) == squeeze_infos_.end()) {
      return false;
    }
    if (IsMultiOutputTensors(func_graph, in_node)) {
      return false;
    }
    if (i == 1) {
      axis = squeeze_infos_[in_cnode].first;
      out_dims = squeeze_infos_[in_cnode].second;
    } else {
      if (squeeze_infos_[in_cnode].first != axis || squeeze_infos_[in_cnode].second != out_dims) {
        return false;
      }
    }
  }
  auto concat_prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_RET(concat_prim != nullptr, false);
  concat_axis_ = concat_prim->GetAttr(ops::kAxis) == nullptr ? 0 : GetValue<int64_t>(concat_prim->GetAttr(ops::kAxis));
  axis = axis < 0 ? axis + out_dims + 1 : axis;
  MS_CHECK_TRUE_RET(axis >= 0 && axis <= out_dims, false);
  concat_axis_ = concat_axis_ < 0 ? concat_axis_ + out_dims : concat_axis_;
  MS_CHECK_TRUE_RET(concat_axis_ >= 0 && concat_axis_ < out_dims, false);
  if (concat_axis_ >= axis) {
    ++concat_axis_;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
