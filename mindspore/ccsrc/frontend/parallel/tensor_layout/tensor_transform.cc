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

#include "frontend/parallel/tensor_layout/tensor_transform.h"
#include <functional>
#include <algorithm>
#include <memory>
#include <utility>
#include <string>
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
const size_t kAllConcatSize = 3;
const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
const size_t kSize1 = 1;
const size_t kSize2 = 2;
const size_t kSize3 = 3;

TensorTransform::TensorTransform() {}

std::shared_ptr<TensorTransform> TensorTransform::GetInstance() {
  static std::shared_ptr<TensorTransform> inst_tensor_transform_ =
    std::shared_ptr<TensorTransform>(new TensorTransform());
  inst_tensor_transform_->InitTransforOperator();
  return inst_tensor_transform_;
}

void TensorTransform::InitTransforOperator() {
  if (inited_function_) {
    return;
  }
  transform_operator_[RESHAPE] = [this](auto op_pair) { return ExtractReshapeOp(op_pair); };
  transform_operator_[ALL_GATHER] = [this](auto op_pair) { return ExtractAllGatherOp(op_pair); };
  transform_operator_[SPLIT] = [this](auto op_pair) { return ExtractSplitOp(op_pair); };
  transform_operator_[CONCAT] = [this](auto op_pair) { return ExtractConcatOp(op_pair); };
  transform_operator_[STRIDEDSLICE] = [this](auto op_pair) { return ExtractStridedSliceOp(op_pair); };
  infer_shape_operator_[RESHAPE] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferReshapeOp(ori_shape, op_pair);
  };
  infer_shape_operator_[ALL_GATHER] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferAllGatherOp(ori_shape, op_pair);
  };
  infer_shape_operator_[STRIDEDSLICE] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferStridedSliceOp(ori_shape, op_pair);
  };
  inited_function_ = true;
}

// return {op_name, dst_shape}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractReshapeOp(const Operator &reshape_op_pair) const {
  auto op_name = reshape_op_pair.first;
  auto op_params = reshape_op_pair.second.second;
  if (op_params.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The reshape has not contains dst_shape.";
  }
  auto shape_value_ptr = op_params.front().first.second;
  auto dst_shape = GetValue<std::vector<int64_t>>(shape_value_ptr);
  return std::make_pair(op_name, dst_shape);
}

// return {op_name, group_ranks + axis}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractAllGatherOp(
  const Operator &allgather_op_pair) const {
  auto op_name = allgather_op_pair.first;
  auto op_attrs = allgather_op_pair.second.first;
  if (op_attrs.size() < kSize2) {
    MS_LOG(INTERNAL_EXCEPTION) << "The allgather has not contains group attrs.";
  }
  auto group_attr = op_attrs[1].second;
  auto group_ranks = GetValue<std::vector<int64_t>>(group_attr);
  // default allgather axis is 0
  group_ranks.push_back(0);
  return std::make_pair(op_name, group_ranks);
}

// return {op_name, [axis, output_num]}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractSplitOp(const Operator &split_op_pair) const {
  auto op_name = split_op_pair.first;
  auto op_attrs = split_op_pair.second.first;
  if (op_attrs.size() < kSize2) {
    MS_LOG(INTERNAL_EXCEPTION) << "The split has not contains output_num attrs.";
  }
  auto axis_attr = op_attrs[0].second;
  auto axis = GetValue<int64_t>(axis_attr);
  auto output_num_attr = op_attrs[1].second;
  auto output_num = GetValue<int64_t>(output_num_attr);
  std::vector<int64_t> attr_list = {axis, output_num};
  return std::make_pair(op_name, attr_list);
}

// return {op_name, [axis]}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractConcatOp(const Operator &concat_op_pair) const {
  auto op_name = concat_op_pair.first;
  auto op_attrs = concat_op_pair.second.first;
  if (op_attrs.size() < 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "The concat has not contains axis attrs.";
  }
  auto axis_attr = op_attrs[0].second;
  auto axis = GetValue<int64_t>(axis_attr);
  std::vector<int64_t> attr_list = {axis};
  return std::make_pair(op_name, attr_list);
}

// return {op_name, begin + end + stride}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractStridedSliceOp(
  const Operator &slice_op_pair) const {
  auto op_name = slice_op_pair.first;
  auto op_params = slice_op_pair.second.second;
  if (op_params.size() < kSize3) {
    MS_LOG(INTERNAL_EXCEPTION) << "The stridedslice op has not contains begin/end/strides.";
  }
  auto begin_value_ptr = op_params[0].first.second;
  auto begin = GetValue<std::vector<int64_t>>(begin_value_ptr);
  auto end_value_ptr = op_params[1].first.second;
  auto end = GetValue<std::vector<int64_t>>(end_value_ptr);
  auto stride_value_ptr = op_params[2].first.second;
  auto stride = GetValue<std::vector<int64_t>>(stride_value_ptr);
  std::vector<int64_t> stride_attr;
  (void)std::copy(begin.begin(), begin.end(), std::back_inserter(stride_attr));
  (void)std::copy(end.begin(), end.end(), std::back_inserter(stride_attr));
  (void)std::copy(stride.begin(), stride.end(), std::back_inserter(stride_attr));
  return std::make_pair(op_name, stride_attr);
}

void TensorTransform::OptimizeAllConcat(std::vector<std::pair<std::string, std::vector<int64_t>>> *transform_op_list) {
  if (transform_op_list->size() < kAllConcatSize) {
    return;
  }
  std::vector<size_t> allconcat_index;
  for (size_t i = kAllConcatSize - 1; i < transform_op_list->size(); ++i) {
    if ((*transform_op_list)[i - kIndex2].first != ALL_GATHER || (*transform_op_list)[i - 1].first != SPLIT ||
        (*transform_op_list)[i].first != CONCAT) {
      continue;
    }
    auto allgather_group_size = SizeToLong((*transform_op_list)[i - kIndex2].second.size() - 1);
    auto split_axis = ((*transform_op_list)[i - kIndex1].second)[kIndex0];
    auto split_size = ((*transform_op_list)[i - kIndex1].second)[kIndex1];
    auto concat_axis = (*transform_op_list)[i].second.front();
    if (allgather_group_size != split_size || split_axis != 0) {
      continue;
    }
    (*transform_op_list)[i - kIndex2].second.back() = concat_axis;
    allconcat_index.push_back(i);
  }
  for (int j = SizeToInt(allconcat_index.size()) - 1; j >= 0; --j) {
    auto erase_index = allconcat_index[IntToSize(j)];
    (void)transform_op_list->erase(transform_op_list->begin() + erase_index);
    (void)transform_op_list->erase(transform_op_list->begin() + erase_index - 1);
  }
}

std::vector<std::pair<std::string, std::vector<int64_t>>> TensorTransform::TransformOperators(const Shapes &from,
                                                                                              const Shapes &to,
                                                                                              const RankList &dev_list,
                                                                                              int64_t rank_id) {
  TensorLayout from_layout;
  (void)from_layout.InitFromVector(from[kIndex0], from[kIndex1], from[kIndex2]);
  TensorLayout to_layout;
  (void)to_layout.InitFromVector(to[kIndex0], to[kIndex1], to[kIndex2]);
  (void)tensor_redistribution_.Init(from_layout, to_layout, dev_list);
  auto origin_rank_id = ParallelContext::GetInstance()->global_rank();
  ParallelContext::GetInstance()->set_do_transform(true);
  ParallelContext::GetInstance()->set_global_rank(rank_id);
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution_.InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Infer tensor redistribution failed.";
  }
  if (redistribution_oplist_ptr->first.size() != redistribution_oplist_ptr->second.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The redistribution op list size cannot match redistribution output info list size.";
  }
  auto operators_vector = redistribution_oplist_ptr->first;
  std::vector<std::pair<std::string, std::vector<int64_t>>> transform_op_list;
  for (auto op_pair : operators_vector) {
    auto op_name = op_pair.first;
    auto it = transform_operator_.find(op_name);
    if (it == transform_operator_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The op:" << op_name << " is not a valid redistrbution op.";
    }
    transform_op_list.push_back(it->second(op_pair));
  }
  OptimizeAllConcat(&transform_op_list);
  ParallelContext::GetInstance()->set_do_transform(false);
  ParallelContext::GetInstance()->set_global_rank(origin_rank_id);
  return transform_op_list;
}

Shape TensorTransform::InferReshapeOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  if (std::accumulate(ori_shape.begin(), ori_shape.end(), 1, std::multiplies<int64_t>()) !=
      std::accumulate(op.begin(), op.end(), 1, std::multiplies<int64_t>())) {
    MS_LOG(EXCEPTION) << "Infer redistribution error, cannot convert shape: " << ori_shape << " to shape:" << op;
  }
  return op;
}

Shape TensorTransform::InferAllGatherOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  auto new_shape = ori_shape;
  auto axis = op.back();
  new_shape[LongToSize(axis)] = new_shape[LongToSize(axis)] * (op.size() - 1);
  return new_shape;
}

Shape TensorTransform::InferStridedSliceOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  size_t end_index = size_t(op.size() / 3);
  if (ori_shape.size() != end_index) {
    MS_LOG(EXCEPTION) << "Infer redistribution error, the shape:" << ori_shape
                      << " cannot be sliced with dimension size:" << end_index;
  }
  auto new_shape = ori_shape;
  for (size_t i = 0; i < ori_shape.size(); ++i) {
    new_shape[i] = (op[end_index + i] - op[i]) / op[kSize2 * end_index + i];
  }
  return new_shape;
}

std::vector<Shape> TensorTransform::GetRedistributionOpShape(
  const Shape &ori_shape, const std::vector<std::pair<std::string, std::vector<int64_t>>> &transform_op_list) {
  std::vector<Shape> result_shape;
  auto cur_shape = ori_shape;
  for (const auto &op : transform_op_list) {
    auto op_name = op.first;
    auto it = infer_shape_operator_.find(op_name);
    if (it == infer_shape_operator_.end()) {
      MS_LOG(EXCEPTION) << "The op:" << op_name << " cannot infer shape in redistribution.";
    }
    cur_shape = it->second(cur_shape, op.second);
    result_shape.push_back(cur_shape);
  }
  return result_shape;
}

Operator ConstructReshapeOp(const std::vector<int64_t> &shape) {
  OperatorAttrs attrs;
  ValuePtr param_value = MakeValue(shape);
  Attr param = std::make_pair(SHAPE, param_value);
  OperatorParams params = {std::make_pair(param, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(RESHAPE, args);
}

RedistributionOpListPtr TensorTransform::OptimizeTensorRedistributionOperatorList(
  const RedistributionOpListPtr &redistribution_op_list, const Shape &input_shape) {
  // 1 operators_vector to transform_op_list
  // 2 allgather->split->concat to allconcat
  MS_EXCEPTION_IF_NULL(redistribution_op_list);
  if ((redistribution_op_list->first).size() != (redistribution_op_list->second).size()) {
    return redistribution_op_list;
  }
  auto operators_vector = redistribution_op_list->first;
  std::vector<std::pair<std::string, std::vector<int64_t>>> transform_op_list;
  for (auto op_pair : operators_vector) {
    auto op_name = op_pair.first;
    auto it = transform_operator_.find(op_name);
    if (it == transform_operator_.end()) {
      MS_LOG(INFO) << "The op:" << op_name << " would not be optimized.";
      return redistribution_op_list;
    }
    transform_op_list.push_back(it->second(op_pair));
  }
  OptimizeAllConcat(&transform_op_list);
  auto shape_list = GetRedistributionOpShape(input_shape, transform_op_list);
  size_t current_allgather_pos_in_origin_list = 0;
  std::unordered_map<size_t, std::vector<int64_t>> left_reshape_op_list;
  std::vector<size_t> allconcat_pos_list;
  // 3 remove the dim which value is 1 for AllConcat
  for (size_t i = 0; i < transform_op_list.size(); ++i) {
    auto trans_op_pair = transform_op_list[i];
    if (trans_op_pair.first != ALL_GATHER) {
      current_allgather_pos_in_origin_list++;
      continue;
    }
    auto axis = transform_op_list[i].second.back();
    if (axis == 0) {
      current_allgather_pos_in_origin_list += kSize3;
      continue;
    }
    if (i == transform_op_list.size() - 1 || transform_op_list[i + 1].first != RESHAPE) {
      current_allgather_pos_in_origin_list += kSize3;
      continue;
    }
    auto src_shape = shape_list[i];
    src_shape[LongToSize(axis)] = src_shape[LongToSize(axis)] / (transform_op_list[i].second.size() - 1);
    auto new_axis = axis;
    auto new_src_shape = src_shape;
    for (int32_t j = axis - 1; j >= 0; --j) {
      if (src_shape[j] != 1) {
        continue;
      }
      new_src_shape.erase(new_src_shape.begin() + j);
      new_axis -= 1;
    }
    MS_LOG(INFO) << "src_shape:" << src_shape << ", new_src_shape:" << new_src_shape << ", axis:" << axis
                 << ", new_axis:" << new_axis;
    if (new_axis != 0) {
      current_allgather_pos_in_origin_list += kSize3;
      continue;
    }
    left_reshape_op_list[current_allgather_pos_in_origin_list] = new_src_shape;
    allconcat_pos_list.push_back(current_allgather_pos_in_origin_list);
    current_allgather_pos_in_origin_list += kSize3;
  }
  // Insert reshape and adjust allgather-split-concat for redistribution_op_list
  std::reverse(allconcat_pos_list.begin(), allconcat_pos_list.end());
  for (auto pos : allconcat_pos_list) {
    // erase split concat
    (void)redistribution_op_list->first.erase(redistribution_op_list->first.begin() + pos + kSize2);
    (void)redistribution_op_list->first.erase(redistribution_op_list->first.begin() + pos + kSize1);
    (void)redistribution_op_list->second.erase(redistribution_op_list->second.begin() + pos + kSize2);
    (void)redistribution_op_list->second.erase(redistribution_op_list->second.begin() + pos + kSize1);
    // insert reshape before allgather
    Operator left_reshape_op = ConstructReshapeOp(left_reshape_op_list[pos]);
    (void)redistribution_op_list->first.insert(redistribution_op_list->first.begin() + pos, left_reshape_op);
    (void)redistribution_op_list->second.insert(redistribution_op_list->second.begin() + pos, {false, 0});
  }
  return redistribution_op_list;
}
}  // namespace parallel
}  // namespace mindspore
