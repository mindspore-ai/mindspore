/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/strided_slice_info.h"

#include <bitset>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
// 1, The mask is a int number, it needs to be converted to binary and reversed.
//    (e.g. the input's dimension is 4, and mask is 2, binary is [0, 0, 1, 0], after reversing: [0, 1, 0, 0])
// 2, If the ith bit of `begin_mask` is set, `begin[i]` is ignored.
// 3, If the ith bit of `end_mask` is set, `end[i]` is ignored.
// 4, If the ith bit of `ellipsis_mask` is set, begin[i]/end[i]/strides[i] replace to `...`, it is not supported now.
// 5, If the ith bit of `new_axis_mask` is set:
//    (e.g. input shape: (A, B, C, D), begin: (0, 0), end: (m, n), strides: (1, 1), new_axis_mask: 2)
//    1) The corresponding position is expanded by one dimension; (input shape:(A, 1, B, C, D))
//    2) Ignore the corresponding position of begin/end/strides; (begin: (0, ig), end: (m, ig), strides: (1, ig))
//    3) The output shape is (m, 1, B, C, D)
// 6, If the ith bit of `shrink_axis_mask` is set, delete that dimension.
//    (e.g. input shape: (A, B, C, D), begin: (0, 0), end: (m, n), strides: (1, 1), shrink_axis_mask: 2,
//     the output shape: (m, C, D)
//    notice: if input is [[1, 2], [3, 4]] and all fetch, but shrink_axis_mask is 1, then the output is [1, 2],
//            so if the ith bit of 'shrink_axis_mask' is set, the dimension can not be split
// 7, If the ith bit of `new_axis_mask` and `shrink_axis_mask` are both set, ignore the ith bit of `shrink_axis_mask`.
// 8, The size of begin/mask/strides must be equal, but it can smaller than input's dimension.
// 9, The mask part exceeding the begin/end/strides length is not effective.
Status StridedSliceInfo::GetMask(const std::string &mask_name, int64_t *mask_value) {
  if (mask_value == nullptr) {
    return FAILED;
  }
  auto mask_iter = attrs_.find(mask_name);
  if (mask_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(mask_iter->second);
    if (mask_iter->second->isa<Int64Imm>()) {
      *mask_value = mask_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of " << mask_name << " is not int64_t";
      return FAILED;
    }
  }

  MS_LOG(INFO) << name_ << ": The attr name: " << mask_name << ", the value is " << *mask_value;
  return SUCCESS;
}

constexpr auto kStridedSliceMaxDims = 8;
static std::vector<bool> Dec2Bin(int64_t mask) {
  auto mask_str = std::bitset<kStridedSliceMaxDims>(mask).to_string();
  std::vector<bool> result;
  (void)std::transform(mask_str.rbegin(), mask_str.rend(), std::back_inserter(result),
                       [](const char &c) { return c == '1'; });
  return result;
}

// If the ith bit of `begin_mask` is set, `begin[i]` is ignored.
// The mask part exceeding the begin length is not effective.
void StridedSliceInfo::ComputeBeginMask() {
  for (size_t i = 0; i < begin_mask_bitmap_.size() && i < begin_.size(); ++i) {
    if (begin_mask_bitmap_[i]) {
      begin_[i] = strides_[i] < 0 ? inputs_shape_[0][i] - 1 : 0;
    }
  }

  if (begin_mask_ != 0) {
    MS_LOG(INFO) << name_ << ": The begin is modified to " << begin_;
  }
}

// If the ith bit of `end_mask` is set, `end[i]` is ignored.
// The mask part exceeding the end length is not effective.
void StridedSliceInfo::ComputeEndMask() {
  for (size_t j = 0; j < end_mask_bitmap_.size() && j < end_.size(); ++j) {
    if (end_mask_bitmap_[j]) {
      end_[j] = strides_[j] < 0 ? -1 : inputs_shape_[0][j];
    }
  }

  if (end_mask_ != 0) {
    MS_LOG(INFO) << name_ << ": The end is modified to " << end_;
  }
}

// If the ith bit of `ellipsis_mask` is set, begin[i]/end[i]/strides[i] replace to `...`, it is not supported now.
void StridedSliceInfo::ComputeEllipsisMask() {
  for (size_t k = 0; k < ellipsis_mask_bitmap_.size() && k < begin_.size(); ++k) {
    if (ellipsis_mask_bitmap_[k]) {
      begin_[k] = 0;
      end_[k] = inputs_shape_[0][k];
      strides_[k] = 1;
    }
  }
}

// If the ith bit of `new_axis_mask` is set:
//    (e.g. input shape: (A, B, C, D), begin: (0, 0, 0, 0), end: (m, n, o, p), strides: (1, 1, 1, 1), new_axis_mask: 2)
//    Here, the size of begin/end/strides is equal to input's dimension through ComplementBeginEndStrides()
//    1) The corresponding position is expanded by one dimension; (input shape:(A, 1, B, C, D))
//    2) Ignore the corresponding position of begin/end/strides;
//       (begin: (0, ig, 0, 0), end: (m, ig, o, p), strides: (1, ig, 1, 1))
//    3) The output shape is (m, 1, o, p, D)
// So, use input_shape_in_process_ to generate a tmp input shape
void StridedSliceInfo::ComputeNewAxisMask() {
  input_shape_in_process_ = Shape(inputs_shape_[0].size(), 0);
  for (size_t l = 0; l < new_axis_mask_bitmap_.size() && l < begin_.size() && l < input_shape_in_process_.size(); ++l) {
    if (new_axis_mask_bitmap_[l]) {
      input_shape_in_process_[l] = 1;
      begin_[l] = 0;
      end_[l] = 1;
      strides_[l] = 1;
    }
  }

  size_t count = 0;
  for (auto &ele : input_shape_in_process_) {
    if (ele != 0) {
      continue;
    }
    ele = inputs_shape_[0][count];
    count++;
  }

  (void)input_shape_in_process_.insert(input_shape_in_process_.end(), inputs_shape_[0].begin() + count,
                                       inputs_shape_[0].end());

  if (new_axis_mask_ != 0) {
    MS_LOG(INFO) << name_ << ": The begin is modified to " << begin_ << ", the end is modified to " << end_
                 << ", the strides is modified to " << strides_ << ", the input shape in process is "
                 << input_shape_in_process_;
  }
}

// If the ith bit of `new_axis_mask` and `shrink_axis_mask` are both set, ignore the ith bit of `shrink_axis_mask`.
void StridedSliceInfo::AdjustShrinkAxisMask() {
  bool flag = false;
  for (size_t i = 0; i < new_axis_mask_bitmap_.size(); ++i) {
    if (new_axis_mask_bitmap_[i]) {
      shrink_axis_mask_bitmap_[i] = false;
      flag = true;
    }
  }
  if (flag) {
    MS_LOG(INFO) << name_ << ": The shrink axis mask is modified to " << shrink_axis_mask_bitmap_;
  }
}

Status StridedSliceInfo::GetAttrs() {
  if (attrs_.size() < STRIDED_SLICE_ATTRS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of attrs small than " << STRIDED_SLICE_ATTRS_SIZE;
    return FAILED;
  }

  if ((GetMask(BEGIN_MASK, &begin_mask_) != SUCCESS) || (GetMask(END_MASK, &end_mask_) != SUCCESS) ||
      (GetMask(ELLIPSIS_MASK, &ellipsis_mask_) != SUCCESS) || (GetMask(NEW_AXIS_MASK, &new_axis_mask_) != SUCCESS) ||
      (GetMask(SHRINK_AXIS_MASK, &shrink_axis_mask_) != SUCCESS)) {
    return FAILED;
  }

  if (ellipsis_mask_ != 0) {
    MS_LOG(ERROR) << name_ << ": It can not support ellipsis_mask now";
    return FAILED;
  }

  // convert mask to bit map
  begin_mask_bitmap_ = Dec2Bin(begin_mask_);
  end_mask_bitmap_ = Dec2Bin(end_mask_);
  ellipsis_mask_bitmap_ = Dec2Bin(ellipsis_mask_);
  new_axis_mask_bitmap_ = Dec2Bin(new_axis_mask_);
  shrink_axis_mask_bitmap_ = Dec2Bin(shrink_axis_mask_);
  MS_LOG(INFO) << name_ << ": The begin mask bitmap is " << begin_mask_bitmap_;
  MS_LOG(INFO) << name_ << ": The end mask bitmap is " << end_mask_bitmap_;
  MS_LOG(INFO) << name_ << ": The ellipsis mask bitmap is " << ellipsis_mask_bitmap_;
  MS_LOG(INFO) << name_ << ": The new axis mask bitmap is " << new_axis_mask_bitmap_;
  MS_LOG(INFO) << name_ << ": The shrink axis mask bitmap is " << shrink_axis_mask_bitmap_;

  // if the ith bit of `new_axis_mask` and `shrink_axis_mask` are both set, ignore the ith bit of `shrink_axis_mask`
  AdjustShrinkAxisMask();

  // get begin/end/strides, the size of begin/mask/strides must be equal, but it can smaller than input's dimension
  if (input_value_.size() != STRIDED_SLICE_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of input value must be " << STRIDED_SLICE_INPUTS_SIZE << ", but got "
                  << input_value_.size();
    return FAILED;
  }
  if ((TransValueSequeueToVector(input_value_[STRIDED_SLICE_BEGIN_INDEX], &begin_) != SUCCESS) ||
      (TransValueSequeueToVector(input_value_[STRIDED_SLICE_END_INDEX], &end_) != SUCCESS) ||
      (TransValueSequeueToVector(input_value_[STRIDED_SLICE_STRIDES_INDEX], &strides_) != SUCCESS)) {
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The begin is " << begin_ << ", the end is " << end_ << ", the stride is " << strides_;

  // handle the masks, it will modify the begin/end/strides, the new begin/end/strides are only used for CheckStrategy()
  ComputeBeginMask();
  ComputeEndMask();
  ComputeEllipsisMask();
  ComputeNewAxisMask();
  // no need to handle shrink axis mask
  auto prim = GetCNodePrimitive(cnode_);
  if (prim->HasAttr(parallel::SKIP_REDISTRIBUTION)) {
    skip_redistribution_ = GetValue<bool>(prim->GetAttr(parallel::SKIP_REDISTRIBUTION));
  }

  return SUCCESS;
}

Status StridedSliceInfo::CheckInputStrategy(const Shape &strategy_value) {
  // change the strategy if the new mask axis is set
  Shape strategy_in_process = Shape(strategy_value.size(), 0);
  for (size_t i = 0; i < new_axis_mask_bitmap_.size() && i < begin_.size() && i < strategy_value.size(); ++i) {
    if (new_axis_mask_bitmap_[i]) {
      strategy_in_process[i] = 1;
    }
  }

  size_t count = 0;
  for (auto &ele : strategy_in_process) {
    if (ele != 0) {
      continue;
    }
    ele = strategy_value[count];
    count++;
  }

  (void)strategy_in_process.insert(strategy_in_process.end(), strategy_value.begin() + count, strategy_value.end());
  MS_LOG(INFO) << name_ << ": The strategy in process is " << strategy_in_process;

  for (size_t j = 0; j < strides_.size(); ++j) {
    if ((strides_[j] != 1) && (strategy_in_process[j] > 1)) {
      MS_LOG(ERROR)
        << name_
        << ": When a certain dimension is split, now does not support that the stride is not 1, the strides is "
        << strides_ << ", the strategy is " << strategy_in_process << ", the index is " << j;
      return FAILED;
    }
  }

  for (size_t k = 0; k < begin_.size(); ++k) {
    bool no_fully_fetch = ((begin_[k] != 0) || (end_[k] < input_shape_in_process_[k]));
    if (no_fully_fetch && (strategy_in_process[k] != 1) && !skip_redistribution_) {
      MS_LOG(ERROR) << name_
                    << ": When a dimension is not fully fetched, the dimension can not be split now, the begin is "
                    << begin_ << ", the end is " << end_ << ", the index is " << k << ", the input shape in process is "
                    << input_shape_in_process_ << ", the strategy in process is " << strategy_in_process;
      return FAILED;
    }
  }

  // if the ith bit of 'shrink_axis_mask' is set, the dimension can not be split
  for (size_t l = 0; l < strategy_in_process.size() && l < shrink_axis_mask_bitmap_.size(); ++l) {
    if (shrink_axis_mask_bitmap_[l] && strategy_in_process[l] != 1) {
      MS_LOG(ERROR) << name_
                    << ": When a dimension is shrunk, the dimension can not be split now, the strategy in process is "
                    << strategy_in_process << ", the shrink axis mask bitmap is " << shrink_axis_mask_bitmap_;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status StridedSliceInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  Dimensions strategy_value = stra[0];
  if (strategy_value.size() < strides_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be larger or equal to the size of strides";
    return FAILED;
  }

  return CheckInputStrategy(strategy_value);
}

Status StridedSliceInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status StridedSliceInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);

  // If the ith bit of `new_axis_mask` is set, the corresponding position is expanded by one dimension, and this
  // dimension need to insert MAP_NONE for output tensor map.
  for (size_t j = 0; j < new_axis_mask_bitmap_.size() && j < begin_.size(); ++j) {
    if (new_axis_mask_bitmap_[j]) {
      (void)tensor_map.insert(tensor_map.cbegin() + j, MAP_NONE);
    }
  }

  // If the ith bit of `shrink_axis_mask` is set, delete that dimension.
  Shape out_tensor_map;
  for (size_t k = 0; k < shrink_axis_mask_bitmap_.size() && k < tensor_map.size(); ++k) {
    if (k < begin_.size() && shrink_axis_mask_bitmap_[k]) {
      continue;
    }
    out_tensor_map.push_back(tensor_map[k]);
  }

  MS_LOG(INFO) << name_ << ": The output tensor map is " << out_tensor_map;
  outputs_tensor_map_.push_back(out_tensor_map);
  return SUCCESS;
}

Status StridedSliceInfo::ChangeCNodeBeginEnd() {
  if (!skip_redistribution_) {
    return SUCCESS;
  }
  auto shard_size = strategy_->GetInputDim()[0];
  auto begin_new = begin_;
  auto end_new = end_;
  for (size_t i = 0; i < shard_size.size(); ++i) {
    begin_new[i] = begin_new[i] / shard_size[i];
    end_new[i] = end_new[i] / shard_size[i];
  }
  auto begin_new_value = MakeValue(begin_new);
  auto end_new_value = MakeValue(end_new);
  auto new_begin_value_node = std::make_shared<ValueNode>(begin_new_value);
  auto new_end_value_node = std::make_shared<ValueNode>(end_new_value);
  cnode_->set_input(INDEX_TWO, new_begin_value_node);
  cnode_->set_input(INDEX_THREE, new_end_value_node);
  return SUCCESS;
}

Status StridedSliceInfo::InferMirrorOps() {
  if (ChangeCNodeBeginEnd() != SUCCESS) {
    return FAILED;
  }
  mirror_ops_.clear();
  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs tensor map is empty";
    return FAILED;
  }
  Shape input_tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(input_tensor_map, &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty.";
    return SUCCESS;
  }

  OperatorVector input_op, begin_op, end_op, strides_op;
  input_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(input_op);
  mirror_ops_.push_back(begin_op);
  mirror_ops_.push_back(end_op);
  mirror_ops_.push_back(strides_op);
  return SUCCESS;
}

// Note: if the batch dimension is not fully fetched, the batch strategy may not work.
std::shared_ptr<Strategies> StridedSliceInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "generate batch parallel strategies failed.";
  }
  split_flag_list_ = {true};
  bool no_fully_fetch = ((begin_[0] != 0) || (end_[0] < input_shape_in_process_[0]));
  if (no_fully_fetch) {
    split_flag_list_ = {false};
  }
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

Status StridedSliceInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> StridedSliceInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input_split(inputs_shape_[0].size(), 1);
  for (size_t i = 0; i < begin_.size(); ++i) {
    bool no_fully_fetch = ((begin_[i] != 0) || (end_[i] < inputs_shape_[0][i]));
    if (no_fully_fetch || (strides_[i] != 1)) {
      input_split[i] = 0;
    }
  }
  Shapes splittable_inputs = {input_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": generate strategies failed";
  }

  return sp_vector;
}

REGISTER(StridedSliceInfo);
}  // namespace parallel
}  // namespace mindspore
