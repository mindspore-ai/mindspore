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
  transform_operator_[RESHAPE] = [=](auto op_pair) { return ExtractReshapeOp(op_pair); };
  transform_operator_[ALL_GATHER] = [=](auto op_pair) { return ExtractAllGatherOp(op_pair); };
  transform_operator_[SPLIT] = [=](auto op_pair) { return ExtractSplitOp(op_pair); };
  transform_operator_[CONCAT] = [=](auto op_pair) { return ExtractConcatOp(op_pair); };
  transform_operator_[STRIDEDSLICE] = [=](auto op_pair) { return ExtractStridedSliceOp(op_pair); };
  inited_function_ = true;
}

// return {op_name, dst_shape}
std::pair<std::string, std::vector<int64_t>> TensorTransform::ExtractReshapeOp(const Operator &reshape_op_pair) const {
  auto op_name = reshape_op_pair.first;
  auto op_params = reshape_op_pair.second.second;
  if (op_params.empty()) {
    MS_LOG(EXCEPTION) << "The reshape has not contains dst_shape.";
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
    MS_LOG(EXCEPTION) << "The allgather has not contains group attrs.";
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
    MS_LOG(EXCEPTION) << "The split has not contains output_num attrs.";
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
    MS_LOG(EXCEPTION) << "The concat has not contains axis attrs.";
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
    MS_LOG(EXCEPTION) << "The stridedslice op has not contains begin/end/strides.";
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
  if (redistribution_oplist_ptr->first.size() != redistribution_oplist_ptr->second.size()) {
    MS_LOG(EXCEPTION) << "The redistribution op list size cannot match redistribution output info list size.";
  }
  auto operators_vector = redistribution_oplist_ptr->first;
  std::vector<std::pair<std::string, std::vector<int64_t>>> transform_op_list;
  for (auto op_pair : operators_vector) {
    auto op_name = op_pair.first;
    auto it = transform_operator_.find(op_name);
    if (it == transform_operator_.end()) {
      MS_LOG(EXCEPTION) << "The op:" << op_name << " is not a valid redistrbution op.";
    }
    transform_op_list.push_back(it->second(op_pair));
  }
  OptimizeAllConcat(&transform_op_list);
  ParallelContext::GetInstance()->set_do_transform(false);
  ParallelContext::GetInstance()->set_global_rank(origin_rank_id);
  return transform_op_list;
}
}  // namespace parallel
}  // namespace mindspore
