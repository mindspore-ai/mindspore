/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/preprocess_dynamic_shape.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "tools/optimizer/common/format_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
int DoStack(const CNodePtr &cnode, const ShapeVector &out_shape, ShapeVector *out_data) {
  MS_ASSERT(cnode != nullptr && out_data != nullptr);
  if (!CheckPrimitiveType(cnode, prim::kPrimStack)) {
    return lite::RET_NOT_SUPPORT;
  }
  if (out_shape.size() != 1 || out_shape.front() <= 0) {
    return lite::RET_NOT_SUPPORT;
  }
  auto origin_inputs = cnode->inputs();
  if (lite::RemoveIfDepend(cnode) != RET_OK) {
    cnode->set_inputs(origin_inputs);
    return lite::RET_NOT_SUPPORT;
  }
  if (lite::RemoveIfMakeTuple(cnode) != RET_OK) {
    cnode->set_inputs(origin_inputs);
    return lite::RET_NOT_SUPPORT;
  }
  RemoveIfMonad(cnode);
  auto current_inputs = cnode->inputs();
  for (size_t i = 1; i < current_inputs.size(); ++i) {
    if (utils::isa<CNode>(current_inputs[i])) {
      out_data->push_back(-1);
      continue;
    }
    lite::DataInfo data_info;
    if (lite::FetchConstData(cnode, i, converter::kFmkTypeMs, &data_info, false) != lite::RET_OK) {
      cnode->set_inputs(origin_inputs);
      MS_LOG(ERROR) << "etch stack's const data failed.";
      return lite::RET_ERROR;
    }
    if (data_info.data_ptr_ == nullptr ||
        (data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
        std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1, std::multiplies<>()) != 1) {
      cnode->set_inputs(origin_inputs);
      return lite::RET_NOT_SUPPORT;
    }
    out_data->push_back(*static_cast<int *>(data_info.data_ptr_));
  }
  cnode->set_inputs(origin_inputs);
  return lite::RET_OK;
}

int ArithmeticInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                         std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Mul should have two inputs.";
    return lite::RET_ERROR;
  }
  const auto &first_shape = in_shapes.front();
  const auto &second_shape = in_shapes[1];
  size_t out_shape_size = first_shape.size() >= second_shape.size() ? first_shape.size() : second_shape.size();
  ShapeVector first_shape_expand;
  for (size_t i = 0; i < (out_shape_size - first_shape.size()); ++i) {
    first_shape_expand.push_back(1);
  }
  (void)first_shape_expand.insert(first_shape_expand.end(), first_shape.begin(), first_shape.end());
  ShapeVector second_shape_expand;
  for (size_t i = 0; i < (out_shape_size - second_shape.size()); ++i) {
    second_shape_expand.push_back(1);
  }
  (void)second_shape_expand.insert(second_shape_expand.end(), second_shape.begin(), second_shape.end());
  ShapeVector out_shape;
  for (size_t i = 0; i < out_shape_size; ++i) {
    if (first_shape_expand[i] == second_shape_expand[i]) {
      out_shape.push_back(first_shape_expand[i]);
      continue;
    }
    if (first_shape_expand[i] == 1) {
      out_shape.push_back(second_shape_expand[i]);
      continue;
    }
    if (second_shape_expand[i] == 1) {
      out_shape.push_back(first_shape_expand[i]);
      continue;
    }
    MS_LOG(INFO) << "Mul cannot determine out-shape.";
    return lite::RET_NOT_SUPPORT;
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int CommonInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  out_shapes->clear();
  (void)out_shapes->insert(out_shapes->begin(), in_shapes.begin(), in_shapes.end());
  return lite::RET_OK;
}

int ConcatInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeTwo || in_shapes.empty()) {
    MS_LOG(ERROR) << "Concat should have at least one input.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "Concat's primitive is a nullptr.");
  int axis = 0;
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    axis = GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  }
  ShapeVector out_shape = in_shapes.front();
  size_t rank = out_shape.size();
  if (axis < 0) {
    axis += rank;
  }
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < static_cast<int>(rank), lite::RET_ERROR,
                    "Concat's axis doesn't match with shape.");
  int64_t axis_sum = 0;
  for (const auto &in_shape : in_shapes) {
    if (in_shape.size() != rank) {
      return lite::RET_NOT_SUPPORT;
    }
    if (in_shape[axis] < 0) {
      axis_sum = -1;
      break;
    }
    axis_sum += in_shape[axis];
  }
  out_shape[axis] = axis_sum;
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int ExpandDimsInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                         std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Expanddims should have two inputs.";
    return lite::RET_ERROR;
  }
  auto second_input = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_ERROR, "Expanddims's second input is a nullptr.");
  if (second_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Expanddims fetch second-input's data failed.");
  MS_CHECK_TRUE_MSG(data_info.data_ptr_ != nullptr, lite::RET_ERROR,
                    "Expanddims's second-input's data shouldn't a nullptr.");
  MS_CHECK_TRUE_MSG(data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32, lite::RET_ERROR,
                    "Expanddims's second-input's data-type should be int.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  MS_CHECK_TRUE_MSG(element_num == 1, lite::RET_ERROR, "Expanddims's second-input should be a scalar.");
  auto axis = *static_cast<int *>(data_info.data_ptr_);
  auto first_shape = in_shapes.front();
  auto first_shape_size = static_cast<int>(first_shape.size());
  if (axis < 0) {
    axis = first_shape_size + axis + 1;
  }
  MS_CHECK_TRUE_MSG(axis >= 0 && axis <= first_shape_size, lite::RET_ERROR, "Expanddims's second-input is invalid.");
  out_shapes->clear();
  (void)first_shape.insert(first_shape.begin() + axis, 1);
  out_shapes->push_back(first_shape);
  return lite::RET_OK;
}

int GatherInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeFour || in_shapes.size() < kInputSizeThree) {
    MS_LOG(ERROR) << "Gther should have three inputs.";
    return lite::RET_ERROR;
  }
  auto third_input = cnode->input(kInputIndexThree);
  MS_CHECK_TRUE_MSG(third_input != nullptr, lite::RET_ERROR, "Gather's third input is a nullptr.");
  if (third_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexThree, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Gather fetch second-input's data failed.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  MS_CHECK_TRUE_MSG(element_num <= 1, lite::RET_ERROR, "Gather's second-input should be a scalar.");
  int axis{0};
  if (element_num == 1) {
    MS_CHECK_TRUE_MSG(data_info.data_ptr_ != nullptr, lite::RET_ERROR,
                      "Gather's second-input's data shouldn't a nullptr.");
    if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
      axis = *static_cast<int *>(data_info.data_ptr_);
    } else if (data_info.data_type_ == kNumberTypeInt64) {
      axis = *static_cast<int64_t *>(data_info.data_ptr_);
    } else {
      MS_LOG(ERROR) << "Gather's axis is invalid, which should be int or int64.";
      return lite::RET_ERROR;
    }
  }
  const auto &first_shape = in_shapes.front();
  auto first_shape_size = static_cast<int>(first_shape.size());
  if (axis < 0) {
    axis = first_shape_size + axis;
  }
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < first_shape_size, lite::RET_ERROR, "Gather's axis out of range.");
  const auto &second_shape = in_shapes[1];
  ShapeVector out_shape;
  for (int i = 0; i < axis; ++i) {
    out_shape.push_back(first_shape[i]);
  }
  (void)out_shape.insert(out_shape.end(), second_shape.begin(), second_shape.end());
  for (int i = axis + 1; i < first_shape_size; ++i) {
    out_shape.push_back(first_shape[i]);
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int MatMulInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "MatMul should have at least two inputs.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "MatMul's primitive is a nullptr.");
  bool a_trans = prim->GetAttr(ops::kTransposeA) && GetValue<bool>(prim->GetAttr(ops::kTransposeA));
  bool b_trnas = prim->GetAttr(ops::kTransposeB) && GetValue<bool>(prim->GetAttr(ops::kTransposeB));
  const auto &a_shape = in_shapes.front();
  MS_CHECK_TRUE_RET(a_shape.size() >= kInputSizeTwo, lite::RET_NOT_SUPPORT);
  const auto &b_shape = in_shapes[1];
  MS_CHECK_TRUE_RET(b_shape.size() >= kInputSizeTwo, lite::RET_NOT_SUPPORT);
  size_t a_rank = a_shape.size();
  size_t b_rank = b_shape.size();
  size_t out_rank = std::max(a_rank, b_rank);
  ShapeVector a_pre_shape;
  (void)a_pre_shape.insert(a_pre_shape.end(), out_rank - a_rank, 1);
  (void)a_pre_shape.insert(a_pre_shape.end(), a_shape.begin(), a_shape.begin() + a_rank - C2NUM);
  ShapeVector b_pre_shape;
  (void)b_pre_shape.insert(b_pre_shape.end(), out_rank - b_rank, 1);
  (void)b_pre_shape.insert(b_pre_shape.end(), b_shape.begin(), b_shape.begin() + b_rank - C2NUM);
  ShapeVector out_shape;
  MS_ASSERT(a_pre_shape.size() == b_pre_shape.size());
  for (size_t i = 0; i < out_rank - C2NUM; ++i) {
    if (a_pre_shape[i] == b_pre_shape[i]) {
      out_shape.push_back(a_pre_shape[i]);
      continue;
    }
    if (a_pre_shape[i] == 1) {
      out_shape.push_back(b_pre_shape[i]);
      continue;
    }
    if (b_pre_shape[i] == 1) {
      out_shape.push_back(a_pre_shape[i]);
      continue;
    }
    return lite::RET_NOT_SUPPORT;
  }
  out_shape.push_back(a_trans ? a_shape.back() : a_shape[a_rank - C2NUM]);
  out_shape.push_back(b_trnas ? b_shape[b_rank - C2NUM] : b_shape.back());
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int ReduceInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Reduce should have two inputs.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "Reduce's primitive is a nullptr.");
  bool keep_dim = prim->GetAttr(ops::kKeepDims) != nullptr && GetValue<bool>(prim->GetAttr(ops::kKeepDims));
  bool reduce_to_end = prim->GetAttr(ops::kReduceToEnd) != nullptr && GetValue<bool>(prim->GetAttr(ops::kReduceToEnd));
  if (reduce_to_end) {
    return lite::RET_NOT_SUPPORT;
  }
  auto second_input = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_ERROR, "Reduce's second input is a nullptr.");
  if (second_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Reduce fetch second-input's data failed.");
  MS_CHECK_TRUE_MSG(data_info.shape_.size() <= 1, lite::RET_ERROR, "Reduce second-input should be <= 1D.");
  std::set<int> reduce_axes;
  int rank = static_cast<int>(in_shapes.front().size());
  if (data_info.data_ptr_ == nullptr) {
    (void)reduce_axes.insert(0);
  } else {
    int element_num = data_info.shape_.empty() ? 1 : data_info.shape_.front();
    std::vector<int> temp;
    int *axes{nullptr};
    if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
      axes = static_cast<int *>(data_info.data_ptr_);
    } else if (data_info.data_type_ == kNumberTypeInt64) {
      (void)temp.insert(temp.begin(), static_cast<int64_t *>(data_info.data_ptr_),
                        static_cast<int64_t *>(data_info.data_ptr_) + element_num);
      axes = temp.data();
    } else {
      return lite::RET_NOT_SUPPORT;
    }
    for (int i = 0; i < element_num; ++i) {
      int axis = axes[i] >= 0 ? axes[i] : axes[i] + rank;
      MS_CHECK_TRUE_MSG(axis >= 0 && axis < rank, lite::RET_ERROR, "Reduce's axis is out of range.");
      (void)reduce_axes.insert(axis);
    }
  }
  int start = 0;
  ShapeVector out_shape;
  for (auto iter = reduce_axes.begin(); iter != reduce_axes.end(); ++iter) {
    int end = *iter;
    for (; start < end; ++start) {
      out_shape.push_back(in_shapes.front()[start]);
    }
    if (keep_dim) {
      out_shape.push_back(1);
    }
    ++start;
  }
  for (; start < rank; ++start) {
    out_shape.push_back(in_shapes.front()[start]);
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int ReshapeInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                      std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeTwo) {
    (void)out_shapes->emplace_back();
    return lite::RET_OK;
  }
  if (in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Reshape should have two inputs.";
    return lite::RET_ERROR;
  }
  out_shapes->clear();
  auto second_input = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_ERROR, "Reshape's second input is a nullptr.");
  if (second_input->isa<CNode>()) {
    const auto &second_in_shape = in_shapes[1];
    if (second_in_shape.size() != 1 || second_in_shape.front() <= 0) {
      return lite::RET_NOT_SUPPORT;
    }
    ShapeVector out_shape;
    auto ret = DoStack(second_input->cast<CNodePtr>(), second_in_shape, &out_shape);
    if (ret == lite::RET_NOT_SUPPORT) {
      out_shape = ShapeVector(second_in_shape.front(), -1);
    } else if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Do stack failed.";
      return ret;
    }
    out_shapes->push_back(out_shape);
    return lite::RET_OK;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Reshape fetch second-input's data failed.");
  MS_CHECK_TRUE_MSG(data_info.shape_.size() <= 1, lite::RET_ERROR, "Reshape second-input should be <= 1D.");
  if (data_info.data_ptr_ == nullptr || (data_info.shape_.size() == 1 && data_info.shape_.front() == 0)) {
    (void)out_shapes->emplace_back();
  }
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  ShapeVector out_shape;
  if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
    for (int i = 0; i < element_num; ++i) {
      out_shape.push_back(*(static_cast<int *>(data_info.data_ptr_) + i));
    }
  } else if (data_info.data_type_ == kNumberTypeInt64) {
    for (int i = 0; i < element_num; ++i) {
      out_shape.push_back(*(static_cast<int64_t *>(data_info.data_ptr_) + i));
    }
  } else {
    return lite::RET_NOT_SUPPORT;
  }
  const auto &in_shape = in_shapes.front();
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0) {
      MS_CHECK_TRUE_MSG(in_shape.size() > i, lite::RET_ERROR, "Reshape's in-rank is invalid.");
      out_shape[i] = in_shape[i];
    }
  }
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int ShapeInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                    std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeTwo || in_shapes.empty()) {
    MS_LOG(ERROR) << "Shape should have one inputs.";
    return lite::RET_ERROR;
  }
  ShapeVector out_shape = {static_cast<int64_t>(in_shapes.front().size())};
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int SplitInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                    std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeTwo || in_shapes.empty()) {
    MS_LOG(ERROR) << "Split should have one inputs.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  auto out_num = prim->GetAttr(ops::kOutputNum) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kOutputNum));
  auto size_splits = prim->GetAttr(ops::kSizeSplits) == nullptr
                       ? std::vector<int64_t>{}
                       : GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kSizeSplits));
  out_num = (out_num == 0 ? static_cast<int64_t>(size_splits.size()) : out_num);
  if (out_num <= 0) {
    return lite::RET_NOT_SUPPORT;
  }
  auto axis = prim->GetAttr(ops::kAxis) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  auto &in_shape = in_shapes.front();
  axis = axis < 0 ? static_cast<int64_t>(in_shape.size()) + axis : axis;
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < static_cast<int64_t>(in_shape.size()), lite::RET_ERROR,
                    "Split's axis is out of range.");
  out_shapes->clear();
  ShapeVector out_shape = in_shape;
  if (size_splits.empty()) {
    MS_CHECK_TRUE_MSG(in_shape[axis] > 0 && in_shape[axis] % out_num == 0, lite::RET_ERROR,
                      "Split's dim doesn't match split-axis.");
    out_shape[axis] = in_shape[axis] / out_num;
    (void)out_shapes->insert(out_shapes->end(), out_num, out_shape);
  } else {
    for (auto v : size_splits) {
      out_shape[axis] = v;
      out_shapes->push_back(out_shape);
    }
  }
  return lite::RET_OK;
}

int SqueezeInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                      std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (in_shapes.empty()) {
    MS_LOG(ERROR) << "Squeeze should have one input at least.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Squeeze's primitive is a nullptr.";
    return lite::RET_ERROR;
  }
  auto axes = prim->GetAttr(ops::kAxis) != nullptr ? GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kAxis))
                                                   : std::vector<int64_t>();
  auto &in_shape = in_shapes.front();
  ShapeVector out_shape;
  if (axes.empty()) {
    for (size_t i = 0; i < in_shape.size(); ++i) {
      if (in_shape[i] < 0) {
        return lite::RET_NOT_SUPPORT;
      }
      if (in_shape[i] != 1) {
        out_shape.push_back(in_shape[i]);
      }
    }
  } else {
    auto dims = static_cast<int64_t>(in_shape.size());
    std::vector<int> flags(dims, 0);
    for (auto axis : axes) {
      axis = axis < 0 ? axis + dims : axis;
      if (axis < 0 || axis >= dims) {
        MS_LOG(ERROR) << "Squeeze's axis is invalid. node name is " << cnode->fullname_with_scope();
        return lite::RET_ERROR;
      }
      flags[axis] = 1;
    }
    for (int64_t i = 0; i < dims; ++i) {
      if (flags[i] == 0) {
        out_shape.push_back(in_shape[i]);
      }
    }
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int StackInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                    std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (in_shapes.empty()) {
    MS_LOG(ERROR) << "Stack should have one input at least.";
    return lite::RET_ERROR;
  }
  auto dims = in_shapes.front().size();
  if (std::any_of(in_shapes.begin(), in_shapes.end(),
                  [dims](const ShapeVector &in_shape) { return in_shape.size() != dims; })) {
    MS_LOG(ERROR) << "Stack all-inputs should hava same rank.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (std::any_of(in_shapes.begin(), in_shapes.end(), [](const ShapeVector &in_shape) {
        return std::any_of(in_shape.begin(), in_shape.end(), [](int64_t val) { return val == 0; });
      })) {
    return lite::RET_NOT_SUPPORT;
  }
  auto prim = GetCNodePrimitive(cnode);
  auto axis = prim->GetAttr(ops::kAxis) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  if (axis < 0) {
    axis += static_cast<int64_t>(dims);
  }
  if (axis < 0 || axis > static_cast<int64_t>(dims)) {
    MS_LOG(ERROR) << "stack's axis is invalid.";
    return lite::RET_PARAM_INVALID;
  }
  ShapeVector out_shape;
  auto FillShape = [&out_shape, &in_shapes](int64_t start, int64_t end) mutable {
    for (; start < end; ++start) {
      ShapeVector vertical;
      for (const auto &in_shape : in_shapes) {
        if (in_shape[start] >= 0) {
          vertical.push_back(in_shape[start]);
        } else if (in_shape[start] != -1) {
          MS_LOG(ERROR) << "Stack's input-shape must not have a dim-value less than -1.";
          return lite::RET_INPUT_TENSOR_ERROR;
        }
      }
      out_shape.push_back(vertical.size() < in_shapes.size() ? -1 : vertical.front());
      if (!vertical.empty()) {
        int64_t dim = vertical.front();
        if (std::any_of(vertical.begin(), vertical.end(), [dim](const int64_t value) { return value != dim; })) {
          MS_LOG(ERROR) << "Stack's input-shape must be same each other.";
          return lite::RET_INPUT_TENSOR_ERROR;
        }
      }
    }
    return lite::RET_OK;
  };
  if (FillShape(0, axis) != lite::RET_OK) {
    MS_LOG(ERROR) << "Stack do fillShape failed.";
    return lite::RET_ERROR;
  }
  out_shape.push_back(static_cast<int64_t>(in_shapes.size()));
  if (FillShape(axis, dims) != lite::RET_OK) {
    MS_LOG(ERROR) << "Stack do fillShape failed.";
    return lite::RET_ERROR;
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int CheckStridedSlice(const CNodePtr &cnode, int64_t in_rank, lite::DataInfo *begins, lite::DataInfo *ends) {
  MS_ASSERT(cnode != nullptr);
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "StridedSlice's primitive is a nullptr.");
  int64_t ellipsis_mask = prim->GetAttr(ops::kEllipsisMask) ? GetValue<int64_t>(prim->GetAttr(ops::kEllipsisMask)) : 0;
  int64_t new_axis_mask = prim->GetAttr(ops::kNewAxisMask) ? GetValue<int64_t>(prim->GetAttr(ops::kNewAxisMask)) : 0;
  if ((ellipsis_mask | new_axis_mask) != 0) {
    return lite::RET_NOT_SUPPORT;
  }
  for (size_t i = C2NUM; i < kInputSizeFive; ++i) {
    MS_CHECK_TRUE_MSG(cnode->input(i) != nullptr, lite::RET_ERROR, "StridedSlice's input is a nullptr.");
    if (utils::isa<CNode>(cnode->input(i))) {
      return lite::RET_NOT_SUPPORT;
    }
  }
  auto BasicCond = [](const lite::DataInfo &data_info) {
    return data_info.data_ptr_ != nullptr &&
           (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32);
  };
  if (lite::FetchConstData(cnode, C2NUM, converter::kFmkTypeMs, begins, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch StridedSlice's begins failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(begins->shape_.size() == C1NUM && begins->shape_.front() <= in_rank && BasicCond(*begins),
                    lite::RET_NOT_SUPPORT);
  if (lite::FetchConstData(cnode, C3NUM, converter::kFmkTypeMs, ends, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch StridedSlice's ends failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(ends->shape_ == begins->shape_ && BasicCond(*ends), lite::RET_NOT_SUPPORT);
  lite::DataInfo strides;
  if (lite::FetchConstData(cnode, C4NUM, converter::kFmkTypeMs, &strides, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch StridedSlice's strides failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(strides.shape_ == begins->shape_ && BasicCond(strides), lite::RET_NOT_SUPPORT);
  for (int i = 0; i < strides.shape_.front(); ++i) {
    if (static_cast<int *>(strides.data_ptr_)[i] != 1) {
      return lite::RET_NOT_SUPPORT;
    }
  }
  return lite::RET_OK;
}

int StridedSliceInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                           std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() != kInputSizeFive || in_shapes.size() != kInputSizeFour) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo begins;
  lite::DataInfo ends;
  auto ret = CheckStridedSlice(cnode, in_shapes.front().size(), &begins, &ends);
  if (ret != lite::RET_OK) {
    return ret;
  }

  auto prim = GetCNodePrimitive(cnode);
  int64_t begin_mask = prim->GetAttr(ops::kBeginMask) ? GetValue<int64_t>(prim->GetAttr(ops::kBeginMask)) : 0;
  int64_t end_mask = prim->GetAttr(ops::kEndMask) ? GetValue<int64_t>(prim->GetAttr(ops::kEndMask)) : 0;
  int64_t shrink_mask =
    prim->GetAttr(ops::kShrinkAxisMask) ? GetValue<int64_t>(prim->GetAttr(ops::kShrinkAxisMask)) : 0;
  const auto &in_shape = in_shapes.front();
  ShapeVector out_shape;
  int index = 0;
  for (; index < begins.shape_.front(); ++index) {
    if (shrink_mask & (1 << index)) {
      continue;
    }
    int b_mask = begin_mask & (1 << index);
    int e_mask = end_mask & (1 << index);
    if (b_mask && e_mask) {
      out_shape.push_back(in_shape[index]);
      continue;
    }
    int64_t begin = static_cast<int *>(begins.data_ptr_)[index];
    int64_t end = static_cast<int *>(ends.data_ptr_)[index];
    if (b_mask) {
      begin = 0;
    }
    if (e_mask) {
      end = in_shape[index];
    }
    if (in_shape[index] > 0) {
      begin += (begin >= 0 ? 0 : in_shape[index]);
      end += (end >= 0 ? 0 : in_shape[index]);
    }
    if (begin < 0 || end < 0 || begin > end) {
      return lite::RET_NOT_SUPPORT;
    }
    out_shape.push_back(end - begin);
  }
  (void)out_shape.insert(out_shape.end(), in_shape.begin() + index, in_shape.end());
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int TransposeInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                        std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  out_shapes->clear();
  if (in_shapes.size() == 1) {
    auto in_shape = in_shapes.front();
    ShapeVector out_shape(in_shape.rbegin(), in_shape.rend());
    out_shapes->push_back(out_shape);
    return lite::RET_OK;
  }
  if (in_shapes.size() != C2NUM) {
    MS_LOG(ERROR) << "Transpose's input should be 1 or 2, now is " << in_shapes.size();
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (utils::isa<CNode>(cnode->input(ops::kInputIndex2))) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  if (lite::FetchConstData(cnode, ops::kInputIndex2, converter::kFmkTypeMs, &data_info, false)) {
    MS_LOG(ERROR) << "Fetch constant info failed, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (data_info.data_ptr_ == nullptr ||
      (data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32)) {
    return lite::RET_NOT_SUPPORT;
  }
  auto num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1, std::multiplies<>());
  auto in_shape = in_shapes.front();
  if (num != static_cast<int>(in_shape.size())) {
    MS_LOG(ERROR) << "Transpose's perm doesn't match with input.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  std::vector<int> visit_flags(num, 0);
  ShapeVector out_shape;
  for (int i = 0; i < num; ++i) {
    auto dim_index = static_cast<int *>(data_info.data_ptr_)[i];
    if (dim_index < 0 || dim_index >= num || visit_flags[dim_index]) {
      MS_LOG(ERROR) << "Transpose's perm is invalid.";
      return lite::RET_INPUT_TENSOR_ERROR;
    }
    visit_flags[dim_index] = 1;
    out_shape.push_back(in_shape[dim_index]);
  }
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}
}  // namespace

int DynamicShapePreprocessor::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  op_shape_infos_.clear();
  auto is_dynamic = CheckIsDynamicModel(func_graph);
  if (!is_dynamic) {
    return lite::RET_NOT_SUPPORT;
  }
  auto ret = ProcessOps(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Preprocess for mul-reduce-fusion failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool DynamicShapePreprocessor::CheckIsDynamicModel(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(graph_input_shape != nullptr);
  auto graph_inputs = func_graph->get_inputs();
  lite::DataInfo data_info;
  bool is_dynamic{false};
  for (auto &input : graph_inputs) {
    if (!utils::isa<Parameter>(input)) {
      continue;
    }
    auto ret = lite::FetchFromDefaultParam(input->cast<ParameterPtr>(), converter::kFmkTypeMs, &data_info, false);
    if (ret != lite::RET_OK) {
      return false;
    }
    ShapeVector shape(data_info.shape_.begin(), data_info.shape_.end());
    is_dynamic = is_dynamic || std::any_of(shape.begin(), shape.end(), [](int64_t v) { return v == -1; });
    op_shape_infos_[input] = std::make_pair(std::vector<ShapeVector>{}, std::vector<ShapeVector>{shape});
  }
  return is_dynamic;
}

int DynamicShapePreprocessor::ProcessOps(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(ops_can_infer != nullptr);
  std::set<std::string> support_ops = {
    prim::kPrimAddFusion->name(),    prim::kPrimActivation->name(), prim::kPrimCast->name(),
    prim::kPrimConcat->name(),       prim::kPrimExpandDims->name(), prim::kPrimGather->name(),
    prim::kPrimMatMulFusion->name(), prim::kPrimMulFusion->name(),  prim::kPrimNotEqual->name(),
    prim::kPrimReduceFusion->name(), prim::kPrimReshape->name(),    prim::kPrimShape->name(),
    prim::kPrimSplit->name(),        prim::kPrimSqueeze->name(),    prim::kPrimStack->name(),
    prim::kPrimStridedSlice->name(), prim::kPrimTranspose->name()};
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      continue;
    }
    auto op_type = prim->name();
    if (support_ops.find(op_type) == support_ops.end()) {
      continue;
    }
    auto origin_inputs = cnode->inputs();
    if (lite::RemoveIfDepend(cnode) != RET_OK) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    if (lite::RemoveIfMakeTuple(cnode) != RET_OK) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    RemoveIfMonad(cnode);
    auto current_inputs = cnode->inputs();
    bool can_infer = std::any_of(current_inputs.begin(), current_inputs.end(), [this](AnfNodePtr &anf_node) {
      return op_shape_infos_.find(anf_node) != op_shape_infos_.end() || !utils::isa<CNode>(anf_node);
    });
    if (!can_infer) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    auto ret = DoInfer(cnode, op_type);
    cnode->set_inputs(origin_inputs);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "error occurred when infer " << op_type;
      return ret;
    }
  }
  return lite::RET_OK;
}

int DynamicShapePreprocessor::DoInfer(const CNodePtr &cnode, const std::string &op_type) {
  MS_ASSERT(cnode != nullptr);
  std::map<std::string, std::function<int(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                                          std::vector<ShapeVector> *out_shapes)>>
    infer_func = {
      {prim::kPrimAddFusion->name(), ArithmeticInferShape},  {prim::kPrimActivation->name(), CommonInferShape},
      {prim::kPrimCast->name(), CommonInferShape},           {prim::kPrimConcat->name(), ConcatInferShape},
      {prim::kPrimExpandDims->name(), ExpandDimsInferShape}, {prim::kPrimGather->name(), GatherInferShape},
      {prim::kPrimMatMulFusion->name(), MatMulInferShape},   {prim::kPrimMulFusion->name(), ArithmeticInferShape},
      {prim::kPrimNotEqual->name(), CommonInferShape},       {prim::kPrimReduceFusion->name(), ReduceInferShape},
      {prim::kPrimReshape->name(), ReshapeInferShape},       {prim::kPrimShape->name(), ShapeInferShape},
      {prim::kPrimSplit->name(), SplitInferShape},           {prim::kPrimSqueeze->name(), SqueezeInferShape},
      {prim::kPrimStack->name(), StackInferShape},           {prim::kPrimStridedSlice->name(), StridedSliceInferShape},
      {prim::kPrimTranspose->name(), TransposeInferShape}};
  if (infer_func.find(op_type) == infer_func.end()) {
    MS_LOG(ERROR) << "Current op: " << op_type << " doesn't support infer.";
    return lite::RET_ERROR;
  }
  std::vector<ShapeVector> in_shapes;
  lite::DataInfo data_info;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    if (input == nullptr) {
      continue;
    }
    if (utils::isa<CNode>(input)) {
      auto real_input_info = GetRealCertainVarInput(cnode, i);
      MS_CHECK_TRUE_MSG(real_input_info.first != nullptr, lite::RET_ERROR, "Current op is invalid.");
      if (op_shape_infos_.find(real_input_info.first) == op_shape_infos_.end()) {
        return lite::RET_OK;
      }
      auto &upper_node_out = op_shape_infos_[real_input_info.first].second;
      auto index = real_input_info.second;
      MS_CHECK_TRUE_MSG(index >= 0 && index < static_cast<int>(upper_node_out.size()), lite::RET_ERROR,
                        "Current op is invalid.");
      in_shapes.push_back(upper_node_out[index]);
    } else {
      auto ret = lite::FetchConstData(cnode, i, converter::kFmkTypeMs, &data_info, false);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Fetch constant info failed, " << cnode->fullname_with_scope();
        return lite::RET_ERROR;
      }
      ShapeVector in_shape(data_info.shape_.begin(), data_info.shape_.end());
      in_shapes.push_back(in_shape);
    }
  }
  auto func = infer_func[op_type];
  MS_ASSERT(func != nullptr);
  std::vector<ShapeVector> out_shapes;
  auto ret = func(cnode, in_shapes, &out_shapes);
  if (ret == lite::RET_NOT_SUPPORT) {
    return lite::RET_OK;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "current op is invalid, " << op_type;
    return lite::RET_ERROR;
  }
  op_shape_infos_[cnode] = std::make_pair(in_shapes, out_shapes);
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
