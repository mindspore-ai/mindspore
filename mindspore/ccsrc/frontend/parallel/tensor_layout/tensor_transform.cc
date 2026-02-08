/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "include/frontend/parallel/tensor_layout/tensor_transform.h"
#include <functional>
#include <algorithm>
#include <memory>
#include <utility>
#include <string>
#include <set>
#include "include/utils/parallel_context.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {
const size_t kAllConcatSize = 3;
const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
const size_t kSize1 = 1;
const size_t kSize2 = 2;
const size_t kSize3 = 3;
const size_t kSize4 = 4;

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
  transform_operator_[ALL_TO_ALL] = [this](auto op_pair) { return ExtractAlltoAllOp(op_pair); };

  infer_shape_operator_[RESHAPE] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferReshapeOp(ori_shape, op_pair);
  };
  infer_shape_operator_[ALL_GATHER] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferAllGatherOp(ori_shape, op_pair);
  };
  infer_shape_operator_[ALL_CONCAT] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferAllConcatOp(ori_shape, op_pair);
  };
  infer_shape_operator_[STRIDEDSLICE] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferStridedSliceOp(ori_shape, op_pair);
  };
  infer_shape_operator_[SLICE] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferSliceOp(ori_shape, op_pair);
  };
  infer_shape_operator_[ALL_TO_ALL] = [this](Shape ori_shape, std::vector<int64_t> op_pair) {
    return InferAlltoAllOp(ori_shape, op_pair);
  };

  construct_op_operator_[RESHAPE] = [this](auto op_pair) { return ConstructReshapeOp(op_pair); };
  construct_op_operator_[ALL_GATHER] = [this](auto op_pair) { return ConstructAllGatherOp(op_pair); };
  construct_op_operator_[SPLIT] = [this](auto op_pair) { return ConstructSplitOp(op_pair); };
  construct_op_operator_[CONCAT] = [this](auto op_pair) { return ConstructConcatOp(op_pair); };
  construct_op_operator_[STRIDED_SLICE] = [this](auto op_pair) { return ConstructStrideSliceOp(op_pair); };
  construct_op_operator_[ALL_TO_ALL] = [this](auto op_pair) { return ConstructAlltoAllOp(op_pair); };
  inited_function_ = true;
}

// return {op_name, dst_shape}
RedisOpPair TensorTransform::ExtractReshapeOp(const Operator &reshape_op_pair) const {
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
RedisOpPair TensorTransform::ExtractAllGatherOp(const Operator &allgather_op_pair) const {
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
RedisOpPair TensorTransform::ExtractSplitOp(const Operator &split_op_pair) const {
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
RedisOpPair TensorTransform::ExtractConcatOp(const Operator &concat_op_pair) const {
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
RedisOpPair TensorTransform::ExtractStridedSliceOp(const Operator &slice_op_pair) const {
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

RankList TensorTransform::ParseRankListFromGroupName(const std::string &group_name) const {
  RankList rank_list;
  std::string rank_str = "";
  std::string rank_list_name = group_name + "-";
  for (size_t i = 0; i < rank_list_name.size(); i++) {
    if (rank_list_name[i] == '-') {
      int64_t rank_id;
      try {
        rank_id = std::stoi(rank_str.c_str());
      } catch (std::invalid_argument &) {
        MS_LOG(EXCEPTION) << "Invalid rank string for a parameter: " << rank_str;
      }
      rank_list.push_back(rank_id);
      rank_str = "";
    } else if (rank_list_name[i] <= '9' && rank_list_name[i] >= '0') {
      rank_str.push_back(rank_list_name[i]);
    } else {
      MS_LOG(EXCEPTION) << "The rank list name cannot convert to rank list: " << rank_list_name;
    }
  }
  return rank_list;
}

RedisOpPair TensorTransform::ExtractAlltoAllOp(const Operator &a2a_op_pair) const {
  auto op_name = a2a_op_pair.first;
  auto op_attrs = a2a_op_pair.second.first;
  if (op_attrs.size() != kSize4) {
    MS_LOG(EXCEPTION) << "The AlltoAll has wrong number of attrs, expected: " << kSize4
                      << ", but got: " << op_attrs.size();
  }
  auto split_count_attr = op_attrs[0].second;
  auto split_count = GetValue<int64_t>(split_count_attr);
  auto split_dim_attr = op_attrs[1].second;
  auto split_dim = GetValue<int64_t>(split_dim_attr);
  auto concat_dim_attr = op_attrs[2].second;
  auto concat_dim = GetValue<int64_t>(concat_dim_attr);
  auto group_name_attr = op_attrs[3].second;
  auto group_name = GetValue<std::string>(group_name_attr);

  RankList rank_list;
  if (virtual_rank_ < 0) {
    rank_list = g_device_manager->FindRankListByHashName(group_name);
  } else {
    rank_list = ParseRankListFromGroupName(group_name);
  }

  std::vector<int64_t> aa = {split_count, split_dim, concat_dim};
  (void)std::copy(rank_list.begin(), rank_list.end(), std::back_inserter(aa));
  return std::make_pair(op_name, aa);
}

// AllGather(rank_list..., 0)->Split(0, split_num)->Concat(axis>0) => AllConcat(rank_list..., axis)
// AllGather(rank_list..., 0) => AllConcat(rank_list..., 0)
Status TensorTransform::TransAllGatherToAllConcat(
  std::vector<std::pair<std::string, std::vector<int64_t>>> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  std::vector<size_t> allconcat_index;
  for (size_t i = 0; i < transform_op_list->size(); ++i) {
    if ((*transform_op_list)[i].first != ALL_GATHER) {
      continue;
    }
    // Match the pattern of AllGather->Split->Concat
    if (i + kSize2 < transform_op_list->size() && (*transform_op_list)[i].first == ALL_GATHER &&
        (*transform_op_list)[i + kSize1].first == SPLIT && (*transform_op_list)[i + kSize2].first == CONCAT) {
      auto allgather_op_inputs = (*transform_op_list)[i].second;  // (group_list..., axis)
      if (allgather_op_inputs.empty()) {
        MS_LOG(DEBUG) << "Transfer AllGather to AllConcat failed. The AllGather's inputs cannot be empty.";
        return FAILED;
      }
      auto split_op_inputs = (*transform_op_list)[i + kIndex1].second;  // (axis, split_num)
      if (split_op_inputs.size() != kSize2) {
        MS_LOG(DEBUG)
          << "Transfer AllGather to AllConcat failed. The size of Split's inputs must be 2. But got inputs: "
          << split_op_inputs;
        return FAILED;
      }
      auto concat_op_inputs = (*transform_op_list)[i + kIndex2].second;  // (axis)
      if (concat_op_inputs.size() != kSize1) {
        MS_LOG(DEBUG)
          << "Transfer AllGather to AllConcat failed. The size of Concat's inputs must be 1. But got inputs: "
          << concat_op_inputs;
        return FAILED;
      }
      auto allgather_group_size = SizeToLong(allgather_op_inputs.size() - kSize1);
      auto split_axis = split_op_inputs.at(0);
      auto split_size = split_op_inputs.at(1);
      auto concat_axis = concat_op_inputs.at(0);
      if (allgather_group_size != split_size || split_axis != 0) {
        MS_LOG(DEBUG) << "For AllConcat op, split_size(" << split_size << ") must be equal to group_size("
                      << allgather_group_size << ") and split_axis(" << split_axis << ") must be 0.";
        return FAILED;
      }
      (*transform_op_list)[i].second.back() = concat_axis;
    } else {
      if ((*transform_op_list)[i].second.back() != 0) {
        MS_LOG(DEBUG) << "For AllGather op, the gather axis must be 0.";
        return FAILED;
      }
    }
    (*transform_op_list)[i].first = ALL_CONCAT;
    allconcat_index.push_back(i);
  }
  for (int j = SizeToInt(allconcat_index.size()) - 1; j >= 0; --j) {
    auto erase_index = allconcat_index[IntToSize(j)];
    if ((*transform_op_list)[erase_index].second.back() != 0) {
      (void)transform_op_list->erase(transform_op_list->begin() + erase_index + kIndex2);
      (void)transform_op_list->erase(transform_op_list->begin() + erase_index + kIndex1);
    }
  }
  return SUCCESS;
}

// AllConcat(rank_list..., axis>0) => AllGather(rank_list..., 0)->Split(0, split_num)->Concat(axis)
// AllConcat(rank_list..., 0) => AllGather(rank_list..., 0)
Status TensorTransform::TransAllConcatToAllGather(
  std::vector<std::pair<std::string, std::vector<int64_t>>> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  int64_t index = SizeToLong(transform_op_list->size()) - 1;
  while (index >= 0) {
    if ((*transform_op_list)[LongToSize(index)].first != ALL_CONCAT) {
      --index;
      continue;
    }
    auto op_pair = (*transform_op_list)[index];
    auto concat_axis = op_pair.second.back();
    if (concat_axis == 0) {
      (*transform_op_list)[LongToSize(index)].first = ALL_GATHER;
    } else {
      auto split_num = op_pair.second.size() - 1;
      (*transform_op_list)[LongToSize(index)].first = ALL_GATHER;
      (*transform_op_list)[LongToSize(index)].second.back() = 0;
      (void)transform_op_list->insert(
        transform_op_list->begin() + index + kIndex1,
        std::pair<std::string, std::vector<int64_t>>{SPLIT, std::vector<int64_t>{0, SizeToLong(split_num)}});

      (void)transform_op_list->insert(
        transform_op_list->begin() + index + kIndex2,
        std::pair<std::string, std::vector<int64_t>>{CONCAT, std::vector<int64_t>{concat_axis}});
    }
    --index;
  }
  return SUCCESS;
}

Status TensorTransform::TransStridedSliceToSlice(const Shape &input_shape,
                                                 std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  auto shape_list = GetRedistributionOpShape(input_shape, *transform_op_list);
  for (size_t i = 0; i < transform_op_list->size(); ++i) {
    auto op_pair = transform_op_list->at(i);
    if (op_pair.first != STRIDED_SLICE) {
      continue;
    }
    auto in_shape = i > 0 ? shape_list[i - 1] : input_shape;
    auto out_shape = shape_list[i];
    auto dim = out_shape.size();
    auto expect_input_size = dim * kSize3;  // start.size + end.size + strides.size

    auto strided_slice_input = op_pair.second;
    if (strided_slice_input.size() != dim * kSize3) {
      MS_LOG(DEBUG) << "The strided slice input is invalid. input dim is " << dim << " and expected input size is "
                    << expect_input_size << ", but got " << strided_slice_input.size()
                    << ". Skip redistribution optimization.";
      return FAILED;
    }
    auto begin = Shape(op_pair.second.begin(), op_pair.second.begin() + dim);
    auto end = Shape(op_pair.second.begin() + dim, op_pair.second.begin() + 2 * dim);
    auto strides = Shape(op_pair.second.begin() + 2 * dim, op_pair.second.end());
    if (std::any_of(strides.begin(), strides.end(), [](int stride) { return stride != 1; })) {
      MS_LOG(DEBUG) << "The strides is not all 1. Skip redistribution optimization.";
      return FAILED;
    }
    int64_t axis = 0;
    while (axis < SizeToLong(dim) && end[axis] - begin[axis] == in_shape[axis]) {
      ++axis;
    }
    int64_t slice_num = in_shape[axis] / (end[axis] - begin[axis]);
    int64_t index = begin[axis] / (end[axis] - begin[axis]);
    (*transform_op_list)[i] = RedisOpPair(SLICE, {axis, slice_num, index});
  }
  return SUCCESS;
}

Status TensorTransform::TransSliceToStridedSlice(const Shape &input_shape,
                                                 std::vector<RedisOpPair> *transform_op_list) {
  auto shape_list = GetRedistributionOpShape(input_shape, *transform_op_list);
  for (size_t i = 0; i < transform_op_list->size(); ++i) {
    auto &op_pair = transform_op_list->at(i);
    if (op_pair.first != SLICE) {
      continue;
    }
    op_pair.first = STRIDED_SLICE;
    auto in_shape = i > 0 ? shape_list[i - 1] : input_shape;
    auto out_shape = shape_list[i];
    auto dim = out_shape.size();

    auto slice_input = op_pair.second;  // [axis, slice_num, index]
    auto axis = slice_input[kIndex0];
    auto slice_num = slice_input[kIndex1];
    auto index = slice_input[kIndex2];
    std::vector<int64_t> begin(dim, 0);
    auto end = in_shape;
    begin[axis] = in_shape[axis] / slice_num * index;
    end[axis] = begin[axis] + in_shape[axis] / slice_num;
    std::vector<int64_t> strides(dim, 1);
    op_pair.second = begin;
    std::copy(end.begin(), end.end(), std::back_inserter(op_pair.second));
    std::copy(strides.begin(), strides.end(), std::back_inserter(op_pair.second));
  }
  return SUCCESS;
}

std::vector<std::pair<std::string, std::vector<int64_t>>> TensorTransform::TransformOperators(
  const Shapes &from, const Shapes &to, const RankList &dev_list, const bool redist_opt, int64_t rank_id) {
  virtual_rank_ = rank_id;
  TensorLayout from_layout;
  (void)from_layout.InitFromVector(from[kIndex0], from[kIndex1], from[kIndex2]);
  TensorLayout to_layout;
  (void)to_layout.InitFromVector(to[kIndex0], to[kIndex1], to[kIndex2]);
  if (!(redist_opt && ParallelContext::GetInstance()->enable_all2all())) {
    ParallelContext::GetInstance()->set_do_transform(true);
  }

  tensor_redistribution_.SetVirtualRank(rank_id);
  (void)tensor_redistribution_.Init(from_layout, to_layout, dev_list);
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution_.InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Infer tensor redistribution failed.";
  }
  if (redistribution_oplist_ptr->first.size() != redistribution_oplist_ptr->second.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The redistribution op list size cannot match redistribution output info list size.";
  }
  if (redist_opt) {
    redistribution_oplist_ptr = OptimizeTensorRedistributionOperatorList(redistribution_oplist_ptr,
                                                                         tensor_redistribution_.input_shape(), rank_id);
  }
  auto operators_vector = redistribution_oplist_ptr->first;
  std::vector<std::pair<std::string, std::vector<int64_t>>> transform_op_list;
  for (const auto &op_pair : operators_vector) {
    auto op_name = op_pair.first;
    auto it = transform_operator_.find(op_name);
    if (it == transform_operator_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The op:" << op_name << " is not a valid redistrbution op.";
    }
    transform_op_list.push_back(it->second(op_pair));
  }
  TransAllGatherToAllConcat(&transform_op_list);
  if (!(redist_opt && ParallelContext::GetInstance()->enable_all2all())) {
    ParallelContext::GetInstance()->set_do_transform(false);
  }
  return transform_op_list;
}

Shape TensorTransform::InferReshapeOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  if (std::find(op.begin(), op.end(), -1) != op.end()) {
    MS_LOG(DEBUG) << "It's dynamic shape. Reshape to " << op;
    return op;
  }
  if (std::find(ori_shape.begin(), ori_shape.end(), -1) != ori_shape.end()) {
    return op;
  }
  if (std::accumulate(ori_shape.begin(), ori_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()) !=
      std::accumulate(op.begin(), op.end(), static_cast<int64_t>(1), std::multiplies<int64_t>())) {
    MS_LOG(EXCEPTION) << "Infer redistribution error, cannot convert shape: " << ori_shape << " to shape:" << op;
  }
  MS_LOG(DEBUG) << "It's static shape. Reshape to " << op;
  return op;
}

Shape TensorTransform::InferAllGatherOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  auto new_shape = ori_shape;
  auto axis = op.back();
  if (new_shape[LongToSize(axis)] != -1) {
    new_shape[LongToSize(axis)] = new_shape[LongToSize(axis)] * (op.size() - 1);
  }
  return new_shape;
}

Shape TensorTransform::InferAllConcatOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  return InferAllGatherOp(ori_shape, op);
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

Shape TensorTransform::InferSliceOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  if (op.size() != kSize3) {
    MS_LOG(EXCEPTION) << "Infer redistribution error, the input size must be 3, but got " << op.size();
  }
  auto axis = op[kIndex0];
  auto slice_num = op[kIndex1];
  if (axis >= SizeToLong(ori_shape.size())) {
    MS_LOG(EXCEPTION) << "Infer redistribution error, the shape:" << ori_shape
                      << " cannot be sliced with dimension size:" << axis;
  }
  auto new_shape = ori_shape;
  new_shape[axis] /= slice_num;
  return new_shape;
}

Shape TensorTransform::InferAlltoAllOp(const Shape &ori_shape, const std::vector<int64_t> &op) const {
  if (op.size() < kSize3) {
    MS_LOG(EXCEPTION) << "Infer shape for AlltoAll failed, the input size must be no less than: " << kSize3
                      << ", but got " << op.size();
  }
  auto split_count = op[kIndex0];
  auto split_dim = op[kIndex1];
  auto concat_dim = op[kIndex2];
  if (split_dim >= SizeToLong(ori_shape.size()) || concat_dim >= SizeToLong(ori_shape.size())) {
    MS_LOG(EXCEPTION) << "Infer shape for AlltoAll failed, the split_dim is: " << split_dim
                      << ", and concat_dim is: " << concat_dim << ", but the shape size is: " << ori_shape.size();
  }
  if (split_count == 0) {
    MS_LOG(EXCEPTION) << "The split_count is 0 in inferring AlltoAll shape.";
  }
  if (split_dim == concat_dim) {
    MS_LOG(EXCEPTION) << "The split_dim and concat_dim is equal in inferring AlltoAll shape.";
  }

  auto new_shape = ori_shape;
  new_shape[split_dim] /= split_count;
  new_shape[concat_dim] *= split_count;
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

Operator TensorTransform::ConstructReshapeOp(const std::vector<int64_t> &inputs) {
  auto shape = inputs;
  OperatorAttrs attrs;
  ValuePtr param_value = MakeValue(shape);
  Attr param = std::make_pair(SHAPE, param_value);
  OperatorParams params = {std::make_pair(param, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(RESHAPE, args);
}

Operator TensorTransform::ConstructAllGatherOp(const std::vector<int64_t> &inputs) {
  RankList rank_list(inputs.begin(), inputs.end() - 1);
  Group group;
  if (virtual_rank_ < 0) {
    if (g_device_manager->CreateGroup(rank_list, &group) != SUCCESS) {
      MS_LOG(EXCEPTION) << "AllGather op: create group failed";
    }
  } else {
    std::vector<Device> dev_list;
    (void)std::transform(rank_list.begin(), rank_list.end(), std::back_inserter(dev_list),
                         [](auto &rank_id) { return Device(rank_id); });
    (void)group.Init(HCCL_WORLD_GROUP, dev_list);
  }

  std::string group_name = group.name();
  ValuePtr attr_value = MakeValue(group_name);
  Attr attr = std::make_pair(GROUP, attr_value);
  auto group_devices = group.GetDevicesList();
  std::vector<int64_t> group_ranks;
  (void)std::transform(group_devices.begin(), group_devices.end(), std::back_inserter(group_ranks),
                       [](const Device &dev) { return dev.rank(); });
  ValuePtr attr_ranks_value = MakeValue(group_ranks);
  Attr attr_ranks = std::make_pair(GROUP_RANKS, attr_ranks_value);
  OperatorAttrs attrs = {attr, attr_ranks};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(ALL_GATHER, args);
}

Operator TensorTransform::ConstructConcatOp(const std::vector<int64_t> &inputs) {
  auto concat_dim = inputs.front();
  ValuePtr attr_value = MakeValue(concat_dim);
  Attr attr = std::make_pair(AXIS, attr_value);
  OperatorAttrs attrs = {attr};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(CONCAT, args);
}

Operator TensorTransform::ConstructSplitOp(const std::vector<int64_t> &inputs) {
  // tensor_shape_ can not be validated here
  auto split_axis = inputs.at(kIndex0);
  auto split_count = inputs.at(kIndex1);
  if (split_count <= 0) {
    MS_LOG(EXCEPTION) << "Invalid split count when construct Split operator!";
  }
  OperatorAttrs attrs;
  ValuePtr attr_value_axis = MakeValue(split_axis);
  Attr attr_axis = std::make_pair(AXIS, attr_value_axis);
  ValuePtr attr_value_split = MakeValue(split_count);
  Attr attr_split = std::make_pair(OUTPUT_NUM, attr_value_split);
  attrs = {attr_axis, attr_split};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(SPLIT, args);
}
Operator TensorTransform::ConstructStrideSliceOp(const std::vector<int64_t> &inputs) {
  auto dims = inputs.size() / 3;
  std::vector<int64_t> begin(inputs.begin(), inputs.begin() + dims);
  std::vector<int64_t> end(inputs.begin() + dims, inputs.begin() + kSize2 * dims);
  std::vector<int64_t> strides(inputs.begin() + kSize2 * dims, inputs.begin() + kSize3 * dims);
  ValuePtr param_begin_value = MakeValue(begin);
  Param param_begin = std::make_pair(std::make_pair(BEGIN, param_begin_value), STRIDED_SLICE_BEGIN_INDEX + 1);
  ValuePtr param_end_value = MakeValue(end);
  Param param_end = std::make_pair(std::make_pair(END, param_end_value), STRIDED_SLICE_END_INDEX + 1);

  ValuePtr param_strides_value = MakeValue(strides);
  Param param_strides = std::make_pair(std::make_pair(STRIDES, param_strides_value), STRIDED_SLICE_STRIDES_INDEX + 1);

  int64_t value = 0;
  ValuePtr begin_mask = MakeValue(value);
  Param param_begin_mask = std::make_pair(std::make_pair(BEGIN_MASK, begin_mask), STRIDED_SLICE_BEGIN_MASK_INDEX + 1);
  ValuePtr end_mask = MakeValue(value);
  Param param_end_mask = std::make_pair(std::make_pair(END_MASK, end_mask), STRIDED_SLICE_END_MASK_INDEX + 1);
  ValuePtr ellipsis_mask = MakeValue(value);
  Param param_ellipsis_mask =
    std::make_pair(std::make_pair(ELLIPSIS_MASK, ellipsis_mask), STRIDED_SLICE_ELLIPSIS_MASK_INDEX + 1);
  ValuePtr new_axis_mask = MakeValue(value);
  Param param_new_axis_mask =
    std::make_pair(std::make_pair(NEW_AXIS_MASK, new_axis_mask), STRIDED_SLICE_NEW_AXIS_MASK_INDEX + 1);
  ValuePtr shrink_axis_mask = MakeValue(value);
  Param param_shrink_axis_mask =
    std::make_pair(std::make_pair(SHRINK_AXIS_MASK, shrink_axis_mask), STRIDED_SLICE_SHRINK_AXIS_MASK_INDEX + 1);

  OperatorParams params = {param_begin,    param_end,           param_strides,       param_begin_mask,
                           param_end_mask, param_ellipsis_mask, param_new_axis_mask, param_shrink_axis_mask};
  OperatorAttrs attrs;
  OperatorArgs op_args = std::make_pair(attrs, params);

  return std::make_pair(STRIDED_SLICE, op_args);
}

Operator TensorTransform::ConstructAlltoAllOp(const std::vector<int64_t> &inputs) {
  if (inputs.size() <= kSize3) {
    MS_LOG(EXCEPTION) << "When constructing AlltoAll, the input size must be greater than: " << kSize3
                      << ", but got: " << inputs.size();
  }
  ValuePtr split_count_value = MakeValue(inputs[0]);
  Attr attr_split_count = std::make_pair(SPLIT_COUNT, split_count_value);
  ValuePtr split_dim_value = MakeValue(inputs[1]);
  Attr attr_split_dim = std::make_pair(SPLIT_DIM, split_dim_value);
  ValuePtr concat_dim_value = MakeValue(inputs[2]);
  Attr attr_concat_dim = std::make_pair(CONCAT_DIM, concat_dim_value);

  RankList rank_list(inputs.begin() + kSize3, inputs.end());
  Group group;
  if (virtual_rank_ < 0) {
    if (g_device_manager->CreateGroup(rank_list, &group) != SUCCESS) {
      MS_LOG(EXCEPTION) << "AlltoAll op: create group failed.";
    }
  } else {
    std::vector<Device> dev_list;
    (void)std::transform(rank_list.begin(), rank_list.end(), std::back_inserter(dev_list),
                         [](auto &rank_id) { return Device(rank_id); });
    DeviceManager tmp_dm;
    auto group_name = tmp_dm.GenerateGroupNameByRanks(rank_list);
    (void)group.Init(group_name, dev_list);
  }

  ValuePtr attr_value = MakeValue(group.name());
  Attr attr_group = std::make_pair(GROUP, attr_value);
  ValuePtr attr_ranks_value = MakeValue(rank_list);
  Attr attr_ranks = std::make_pair(GROUP_RANKS, attr_ranks_value);

  OperatorAttrs attrs = {attr_split_count, attr_split_dim, attr_concat_dim, attr_group};  // attr_ranks
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  return std::make_pair(ALL_TO_ALL, args);
}

void TensorTransform::ShowRedisOpList(const Shape &input_shape, const std::vector<RedisOpPair> &transform_op_list) {
  auto shape_list = GetRedistributionOpShape(input_shape, transform_op_list);
  size_t idx = 0;
  for (const auto &op_pair : transform_op_list) {
    MS_LOG(DEBUG) << "op_list[" << idx << "] op_name: " << op_pair.first << ", input: " << op_pair.second
                  << ", shape: " << shape_list[idx];
    ++idx;
  }
}

// Eliminate Useless reshape
// Reshape->Reshape to Reshape
// Eliminate reshape if in_shape == out_shape
void TensorTransform::EliminateRedundancyReshape(const Shape &input_shape,
                                                 std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  size_t left_reshape_index = 0;
  while (left_reshape_index < transform_op_list->size()) {
    if (transform_op_list->at(left_reshape_index).first != RESHAPE) {
      ++left_reshape_index;
      continue;
    }
    auto right_reshape_index = left_reshape_index + 1;
    while (right_reshape_index < transform_op_list->size() &&
           transform_op_list->at(right_reshape_index).first == RESHAPE) {
      ++right_reshape_index;
    }
    auto shape_list = GetRedistributionOpShape(input_shape, *transform_op_list);
    auto in_shape = left_reshape_index > 0 ? shape_list[left_reshape_index - 1] : input_shape;
    auto out_shape = shape_list[right_reshape_index - 1];
    (void)transform_op_list->erase(transform_op_list->begin() + left_reshape_index,
                                   transform_op_list->begin() + right_reshape_index);
    if (in_shape != out_shape) {
      (void)transform_op_list->insert(transform_op_list->begin() + left_reshape_index, RedisOpPair{RESHAPE, out_shape});
    }
    ++left_reshape_index;
  }
}

// Merge AllConcat if concat axis are the same and rank is increase
// e.g. AllConcat(0, 2, 0)->AllConcat(0, 4, 0) => AllConcat(0, 2, 4, 6, 0)
// e.g. AllConcat(0, 1, 2, 3, 0)->AllConcat(0, 4, 0) => AllConcat(0, 1, 2, 3, 4, 5, 6, 7, 0)
void TensorTransform::MergeAllConcat(std::vector<RedisOpPair> *transform_op_list) {
  auto is_increase_concat = [](const RedisOpPair &a, const RedisOpPair &b) {
    if (b.first != ALL_CONCAT) {
      return false;
    }
    auto concat_axis_a = a.second.back();
    auto concat_axis_b = b.second.back();
    if (concat_axis_a != concat_axis_b) {
      return false;
    }
    auto concat_interval_a = a.second.at(1) - a.second.at(0);
    auto concat_size_a = SizeToLong(a.second.size()) - 1;
    auto concat_interval_b = b.second.at(1) - b.second.at(0);
    return (concat_interval_a * concat_size_a) == concat_interval_b;
  };
  auto merge_concat_group = [](const RedisOpPair &a, const RedisOpPair &b) {
    RankList rank_list_a(a.second.begin(), a.second.end() - 1);
    RankList rank_list_b(b.second.begin(), b.second.end() - 1);
    std::set<int64_t> common_rank_set;
    std::set<int64_t> rank_set_a(rank_list_a.begin(), rank_list_a.end());
    for (auto rank : rank_list_b) {
      if (rank_set_a.find(rank) != rank_set_a.end()) {
        common_rank_set.insert(rank);
      }
    }
    if (common_rank_set.size() != 1) {
      MS_LOG(EXCEPTION) << "Failed to merge group, because they don't only have a common rank. The first group is "
                        << rank_list_a << ", and the seconde group is " << rank_list_b;
    }
    auto common_rank = *common_rank_set.begin();
    auto concat_interval_a = rank_list_a.at(1) - rank_list_a.at(0);
    auto concat_interval_b = rank_list_b.at(1) - rank_list_b.at(0);
    auto concat_size_b = SizeToLong((rank_list_b.size()));
    auto start = rank_list_b.at(0) - (common_rank - rank_list_a.at(0));
    auto end = start + concat_interval_b * concat_size_b;
    auto step = concat_interval_a;
    std::vector<int64_t> new_input;
    new_input.reserve((end - start) / step + 1);
    for (int64_t i = start; i < end; i += step) {
      new_input.push_back(i);
    }
    new_input.push_back(a.second.back());
    return new_input;
  };
  auto is_all_concat_with_virtual_rank_group = [&](const RedisOpPair &redis_op_pair) {
    if (virtual_rank_ < 0 || redis_op_pair.first != ALL_CONCAT) {
      return false;
    }
    auto redis_op_inputs = redis_op_pair.second;
    RankList rank_list(redis_op_inputs.begin(), redis_op_inputs.begin() + redis_op_inputs.size() - 1);
    auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
    auto virtual_rank_start = virtual_rank_ - virtual_rank_ % interleaved_num;
    auto virtual_rank_end = virtual_rank_start + interleaved_num;
    RankList virtual_rank_list;
    for (int64_t virtual_rank = virtual_rank_start; virtual_rank < virtual_rank_end; ++virtual_rank) {
      virtual_rank_list.push_back(virtual_rank);
    }
    return rank_list == virtual_rank_list;
  };

  size_t concat_index = 0;
  while (concat_index < transform_op_list->size()) {
    if (transform_op_list->at(concat_index).first != ALL_CONCAT ||
        is_all_concat_with_virtual_rank_group(transform_op_list->at(concat_index))) {
      ++concat_index;
      continue;
    }
    while (concat_index + 1 < transform_op_list->size() &&
           is_increase_concat(transform_op_list->at(concat_index), transform_op_list->at(concat_index + 1))) {
      if (is_all_concat_with_virtual_rank_group(transform_op_list->at(concat_index + 1))) {
        break;
      }
      transform_op_list->at(concat_index).second =
        merge_concat_group(transform_op_list->at(concat_index), transform_op_list->at(concat_index + 1));
      (void)transform_op_list->erase(transform_op_list->begin() + concat_index + kIndex1,
                                     transform_op_list->begin() + concat_index + kIndex2);
    }
    ++concat_index;
  }
}

// Merge Slice if pre_slice and next_slice axis is same, then
// Slice(axis, slice_num1, index1)->Slice(axis, slice_num2, index2) =>
// Slice(axis, slice_num1 * slice_num2, index1 * slice_num2 + index2)
// e.g. Slice(0, 4, 2)->Slice(0, 2, 1) => Slice(0, 8, 3)
void TensorTransform::MergeSlice(std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  size_t pre_slice_index = 0;
  while (pre_slice_index < transform_op_list->size()) {
    if (transform_op_list->at(pre_slice_index).first != SLICE) {
      ++pre_slice_index;
      continue;
    }
    size_t next_slice_index = pre_slice_index + 1;
    while (next_slice_index < transform_op_list->size()) {
      if (transform_op_list->at(next_slice_index).first == RESHAPE) {
        ++next_slice_index;
        continue;
      }
      if (transform_op_list->at(next_slice_index).first != SLICE) {
        break;
      }
      auto pre_slice_input = transform_op_list->at(pre_slice_index).second;
      auto next_slice_input = transform_op_list->at(next_slice_index).second;
      if (pre_slice_input[kIndex0] != next_slice_input[kIndex0]) {
        break;
      }
      auto new_axis = pre_slice_input[kIndex0];
      auto new_slice_num = pre_slice_input[kIndex1] * next_slice_input[kIndex1];
      auto new_index = pre_slice_input[kIndex2] * next_slice_input[kIndex1] + next_slice_input[kIndex2];
      std::vector<int64_t> new_slice_input = {new_axis, new_slice_num, new_index};
      (void)transform_op_list->erase(transform_op_list->begin() + pre_slice_index,
                                     transform_op_list->begin() + next_slice_index);
      transform_op_list->at(pre_slice_index).second = new_slice_input;
      next_slice_index = pre_slice_index + 1;
    }
    ++pre_slice_index;
  }
}

// Optimize transform_op_list. If axis > 1 and in_shape[0:axis] == 1, then
// AllConcat(rank_list..., axis > 1) => Reshape->AllConcat(rank_list, axis=0)->Reshape(out_shape)
void TensorTransform::OptimizeAllConcat(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  for (size_t i = 0; i < transform_op_list->size(); ++i) {
    auto op_pair = transform_op_list->at(i);
    auto op_name = op_pair.first;
    if (op_name != ALL_CONCAT) {
      continue;
    }
    auto shape_list = GetRedistributionOpShape(input_shape, *transform_op_list);
    auto in_shape = i > 0 ? shape_list[i - 1] : input_shape;
    auto concat_axis = op_pair.second.back();
    if (concat_axis == 0 ||
        std::any_of(in_shape.begin(), in_shape.begin() + concat_axis, [](int64_t dim) { return dim != 1; })) {
      continue;
    }
    auto pre_shape = Shape(in_shape.begin() + concat_axis, in_shape.end());
    for (int64_t pre_node_index = SizeToLong(i) - 1; pre_node_index >= 0; --pre_node_index) {
      if (transform_op_list->at(pre_node_index).first != RESHAPE) {
        pre_shape = shape_list[pre_node_index];
        break;
      }
    }
    auto reshape_before = RedisOpPair{RESHAPE, pre_shape};
    auto reshape_after = RedisOpPair{RESHAPE, {shape_list[i]}};
    auto opt_all_concat_input = op_pair.second;
    opt_all_concat_input[opt_all_concat_input.size() - 1] = 0;
    auto opt_all_concat = RedisOpPair{ALL_CONCAT, {opt_all_concat_input}};
    (void)transform_op_list->erase(transform_op_list->begin() + i);
    (void)transform_op_list->insert(transform_op_list->begin() + i, reshape_before);
    (void)transform_op_list->insert(transform_op_list->begin() + i + kIndex1, opt_all_concat);
    (void)transform_op_list->insert(transform_op_list->begin() + i + kIndex2, reshape_after);
  }
}

// If axis > 0 and input_shape[0:axis] == 1, then
// Slice(axis>0)->Reshape => Slice(axis=0)->Reshape
void TensorTransform::OptimizeSlice(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  for (int64_t i = SizeToLong(transform_op_list->size()) - 2; i >= 0; --i) {
    auto slice_op_pair = transform_op_list->at(i);
    auto reshape_op_pair = transform_op_list->at(i + 1);
    if (slice_op_pair.first != SLICE || reshape_op_pair.first != RESHAPE) {
      continue;
    }
    auto shape_list = GetRedistributionOpShape(input_shape, *transform_op_list);
    auto in_shape = i > 0 ? shape_list[i - 1] : input_shape;
    auto axis = slice_op_pair.second[kIndex0];
    if (std::any_of(in_shape.begin(), in_shape.begin() + axis, [](int64_t dim) { return dim != 1; })) {
      continue;
    }
    auto new_slice_op_pair = slice_op_pair;
    new_slice_op_pair.second[0] = 0;
    auto new_reshape_op_pair = reshape_op_pair;
    new_reshape_op_pair.second[0] *= slice_op_pair.second[1];
    transform_op_list->at(i) = new_reshape_op_pair;
    transform_op_list->at(i + 1) = new_slice_op_pair;
  }
}

Status TensorTransform::ReorderAndMergeRedistributionOp(const Shape &input_shape,
                                                        std::vector<RedisOpPair> *transform_op_list) {
  MS_EXCEPTION_IF_NULL(transform_op_list);
  // 1. Preprocess for transform_op_list
  // 1.1 Validate, only solve AllConcat, SLICE and Reshape
  std::vector<std::string> valid_op = {ALL_CONCAT, SLICE, RESHAPE};
  if (std::any_of(transform_op_list->begin(), transform_op_list->end(), [&valid_op](const RedisOpPair &op_pair) {
        return std::find(valid_op.begin(), valid_op.end(), op_pair.first) == valid_op.end();
      })) {
    MS_LOG(DEBUG) << "Each transform op in transform_op_list must be the one of " << valid_op;
    return FAILED;
  }
  // 1.2 Eliminate redundancy reshape op
  EliminateRedundancyReshape(input_shape, transform_op_list);
  // 2. Reorder AllConcat and Slice op
  auto cmp = [](const RedisOpPair &a, const RedisOpPair &b) {
    if (a.first == RESHAPE || b.first == RESHAPE) {
      return false;
    }
    if (a.first != b.first) {
      return a.first == ALL_CONCAT;
    }
    if (a.first == ALL_CONCAT) {
      return a.second.back() > b.second.back();
    }
    return a.second.front() < b.second.front();
  };
  std::stable_sort(transform_op_list->begin(), transform_op_list->end(), cmp);
  // 3. Optimize transform_op_list
  // 3.1 Optimize AllConcat
  // 3.2 Optimize Slice
  OptimizeAllConcat(input_shape, transform_op_list);
  OptimizeSlice(input_shape, transform_op_list);
  // 4. Eliminate Useless reshape
  EliminateRedundancyReshape(input_shape, transform_op_list);
  // 5. Merge AllConcat if concat axis are the same and rank is increase
  MergeAllConcat(transform_op_list);
  // 6. Merge Slice if pre_slice and next_slice axis is same
  MergeSlice(transform_op_list);
  return SUCCESS;
}

RedistributionOpList TensorTransform::ConstructRedistributionOpListByRedisOpList(
  const std::vector<RedisOpPair> &transform_op_list) {
  OperatorVector op_vector;
  OutPutInfoVector op_info;
  for (const auto &op_pair : transform_op_list) {
    auto op_name = op_pair.first;
    if (construct_op_operator_.find(op_name) == construct_op_operator_.end()) {
      MS_LOG(EXCEPTION) << "Construct operator failed. Please implement the construct function of op " << op_name;
    }
    (void)op_vector.emplace_back(construct_op_operator_[op_name](op_pair.second));
    if (op_name == SPLIT) {
      (void)op_info.emplace_back(
        std::pair<bool, uint64_t>{true, op_pair.second.back()});  // op_pair.second = {split_axis, split_num}
    } else {
      (void)op_info.emplace_back(std::pair<bool, uint64_t>{false, 0});
    }
  }
  return RedistributionOpList{op_vector, op_info};
}

RedistributionOpListPtr TensorTransform::OptimizeTensorRedistributionOperatorList(
  const RedistributionOpListPtr &redistribution_op_list, const Shape &input_shape, int64_t virtual_rank) {
  MS_LOG(DEBUG) << "Do optimization for tensor redistributions.";
  virtual_rank_ = virtual_rank;
  // 1 operators_vector to transform_op_list
  // 2 allgather->split->concat to allconcat
  MS_EXCEPTION_IF_NULL(redistribution_op_list);
  if ((redistribution_op_list->first).size() != (redistribution_op_list->second).size()) {
    return redistribution_op_list;
  }
  auto operators_vector = redistribution_op_list->first;
  std::vector<std::pair<std::string, std::vector<int64_t>>> transform_op_list;
  for (const auto &op_pair : operators_vector) {
    auto op_name = op_pair.first;
    auto it = transform_operator_.find(op_name);
    if (it == transform_operator_.end() || IsToBeInsertedSplitOp(op_pair)) {
      MS_LOG(INFO) << "The op:" << op_name << " would not be optimized.";
      return redistribution_op_list;
    }
    transform_op_list.push_back(it->second(op_pair));
  }
  if (transform_op_list.size() <= kSize1) {
    return redistribution_op_list;
  }

  if (TransAllGatherToAllConcat(&transform_op_list) != SUCCESS) {
    MS_LOG(DEBUG) << "TransAllGatherToAllConcat failed, stop and skip optimization.";
    return redistribution_op_list;
  }
  if (TransStridedSliceToSlice(input_shape, &transform_op_list) != SUCCESS) {
    MS_LOG(DEBUG) << "TransStridedSliceToSlice failed, stop and skip optimization.";
    return redistribution_op_list;
  }

  MS_LOG(DEBUG) << "===Origin transform operator list===";
  ShowRedisOpList(input_shape, transform_op_list);
  // Optimize the redistribution by merge AllConcat and Slice
  if (ReorderAndMergeRedistributionOp(input_shape, &transform_op_list) != SUCCESS) {
    MS_LOG(DEBUG) << "ReorderAndMergeRedistributionOp failed, stop and skip optimization.";
    return redistribution_op_list;
  }
  MS_LOG(DEBUG) << "===After optimize transform operator list===";
  ShowRedisOpList(input_shape, transform_op_list);

  if (TransSliceToStridedSlice(input_shape, &transform_op_list) != SUCCESS) {
    MS_LOG(DEBUG) << "TransSliceToStridedSlice failed, stop and skip optimization.";
    return redistribution_op_list;
  }
  if (TransAllConcatToAllGather(&transform_op_list) != SUCCESS) {
    MS_LOG(DEBUG) << "TransAllConcatToAllGather failed, stop and skip optimization.";
    return redistribution_op_list;
  }

  auto opt_redistribution_op_list = ConstructRedistributionOpListByRedisOpList(transform_op_list);
  redistribution_op_list->first = opt_redistribution_op_list.first;
  redistribution_op_list->second = opt_redistribution_op_list.second;
  return redistribution_op_list;
}
}  // namespace parallel
}  // namespace mindspore
