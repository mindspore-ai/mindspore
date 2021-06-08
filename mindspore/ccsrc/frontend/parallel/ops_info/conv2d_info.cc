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

#include "frontend/parallel/ops_info/conv2d_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
int64_t Conv2DInfo::GetIntAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attribution of " << attr_name;
    return -1;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The value of " << attr_name << " is not int";
    return -1;
  }

  return attr_iter->second->cast<Int64ImmPtr>()->value();
}

std::string Conv2DInfo::GetStringAttr(const std::string &attr_name) {
  std::string string_attr;
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attribution of " << attr_name;
    return string_attr;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<StringImm>()) {
    MS_LOG(ERROR) << name_ << ": The value of " << attr_name << " is not string";
    return string_attr;
  }

  string_attr = attr_iter->second->cast<StringImmPtr>()->value();
  return string_attr;
}

std::vector<int64_t> Conv2DInfo::GetTupleAttr(const std::string &attr_name) {
  std::vector<int64_t> tuple_attr;
  auto tuple_attr_iter = attrs_.find(attr_name);
  if (tuple_attr_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attribution of " << attr_name;
    return tuple_attr;
  }

  MS_EXCEPTION_IF_NULL(tuple_attr_iter->second);
  tuple_attr = GetValue<std::vector<int64_t>>(tuple_attr_iter->second);

  return tuple_attr;
}

Status Conv2DInfo::GetAttrs() {
  // out_channel
  out_channel_ = GetIntAttr(OUT_CHANNEL);
  if (out_channel_ <= 0) {
    MS_LOG(ERROR) << name_ << ": The attr of out_channel is invalid";
    return FAILED;
  }

  // kernel_size
  auto kernel_size_iter = attrs_.find(KERNEL_SIZE);
  if (kernel_size_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attribution of " << KERNEL_SIZE;
    return FAILED;
  }

  MS_EXCEPTION_IF_NULL(kernel_size_iter->second);
  if (kernel_size_iter->second->isa<Int64Imm>()) {
    int64_t kernel_size = kernel_size_iter->second->cast<Int64ImmPtr>()->value();
    kernel_size_ = {kernel_size, kernel_size};
  } else if (kernel_size_iter->second->isa<ValueTuple>() || kernel_size_iter->second->isa<ValueList>()) {
    kernel_size_ = GetValue<std::vector<int64_t>>(kernel_size_iter->second);
    if (kernel_size_.size() != 2) {
      MS_LOG(ERROR) << name_ << ": The size of kernel_size'tuple must be 2, but got " << kernel_size_.size();
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": The kernel_size must be int or tuple";
    return FAILED;
  }

  // mode
  mode_ = GetIntAttr(MODE);
  if (mode_ != 1) {
    MS_LOG(ERROR) << name_ << ": The mode must be 1, but got " << mode_;
    return FAILED;
  }

  // pad_mode
  pad_mode_ = GetIntAttr(PAD_MODE);
  if (pad_mode_ < 0 || pad_mode_ > 2) {
    MS_LOG(ERROR) << name_ << ": The pad_mode must be in the range of [0, 2], but got " << pad_mode_;
    return FAILED;
  }

  // pad_list
  pad_list_ = GetTupleAttr(PAD_LIST);
  if (pad_list_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of pad_list must be 4, but got " << pad_list_.size();
    return FAILED;
  }

  // stride
  stride_ = GetTupleAttr(STRIDE);
  if (stride_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of stride must be 4, but got " << stride_.size();
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of stride must be 1, but got (" << stride_[0] << ", "
                  << stride_[1] << ")";
    return FAILED;
  }

  // dilation
  dilation_ = GetTupleAttr(DILATION);
  if (dilation_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of dilation must be 4, but got " << dilation_.size();
    return FAILED;
  }

  // group
  group_ = GetIntAttr(GROUP);
  if (group_ != 1) {
    MS_LOG(ERROR) << name_ << ": The group must be 1, but got " << group_;
    return FAILED;
  }

  // format
  format_ = GetStringAttr(FORMAT);
  if (format_ != NCHW) {
    MS_LOG(ERROR) << name_ << ": The format must be 'NCHW', but got " << format_;
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The out channel is " << out_channel_ << ", kernel size is " << kernel_size_
               << ", mode is " << mode_ << ", pad mode is " << pad_mode_ << ", pad list is " << pad_list_
               << ", stride is " << stride_ << ", dilation is " << dilation_ << ", group is " << group_
               << ", format is " << format_;

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (pad_mode_ == 0) {  // 'pad' mode
    MS_LOG(ERROR) << name_ << ": The 'pad' mode do not support to split H or W";
    return FAILED;
  }

  if (pad_mode_ == 1) {  // 'same' mode
    if ((kernel_size_[0] > stride_[2] || kernel_size_[1] > stride_[3]) && h_strategy > 1) {
      MS_LOG(ERROR) << name_ << ": The 'same' mode do not support to split H when kernel_size > stride";
      return FAILED;
    }

    if (kernel_size_[0] <= stride_[2] || kernel_size_[1] <= stride_[3]) {
      int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
      int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;
      if (h_slice_shape % stride_[2] != 0 || w_slice_shape % stride_[3] != 0) {
        MS_LOG(ERROR) << name_
                      << ": The 'same' mode do not support to split H or W when kernel_size <= stride but slice shape "
                         "is not divisible by stride ";
        return FAILED;
      }
    }
  }

  if (pad_mode_ == 2) {  // 'valid' mode
    if ((kernel_size_[0] > stride_[2] && h_strategy > 1) || (kernel_size_[1] > stride_[3] && w_strategy > 1)) {
      MS_LOG(ERROR) << name_ << ": The 'valid' mode do not support to split H or W when kernel_size > stride";
      return FAILED;
    }

    if (kernel_size_[0] <= stride_[2]) {
      int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
      if (h_slice_shape % stride_[2] != 0) {
        MS_LOG(ERROR) << name_
                      << ": The 'valid' mode do not support to split H when kernel_size <= stride but slice shape is "
                         "not divisible by stride ";
        return FAILED;
      }
    }

    if (kernel_size_[1] <= stride_[3]) {
      int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;
      if (w_slice_shape % stride_[3] != 0) {
        MS_LOG(ERROR) << name_
                      << ": The 'valid' mode do not support to split W when kernel_size <= stride but slice shape is "
                         "not divisible by stride ";
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy.size() != 4 || weight_strategy.size() != 4) {
    MS_LOG(ERROR) << name_
                  << ": The size of input strategy or weight strategy must be 4, but the size of input strategy is "
                  << input_strategy.size() << ", the size of weight strategy is " << weight_strategy.size();
    return FAILED;
  }

  if (input_strategy[1] != weight_strategy[1]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-in for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-in for weight strategy is " << weight_strategy[1];
    return FAILED;
  }

  if (weight_strategy[2] != 1 || weight_strategy[3] != 1) {
    MS_LOG(ERROR) << name_ << ": The kernel size can not be split, but the strategy for kernel size is ("
                  << weight_strategy[2] << ", " << weight_strategy[3] << ")";
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  if (weight_strategy[0] > 1) {
    out_channel_shard_ = true;
    new_out_channel_ = out_channel_ / weight_strategy[1];
  } else {
    out_channel_shard_ = false;
  }

  return SUCCESS;
}

Status Conv2DInfo::InferDevMatrixShape() {
  // the strategy is ((n, i, h, w), (o, i, 1, 1))
  // the dev matrix is (n, i, h, w, o)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << "The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  dev_matrix_shape_.push_back(stra[1][0]);
  return SUCCESS;
}

Status Conv2DInfo::InferTensorMap() {
  // input_strategy: ((n, i, h, w), (o, i, 1, 1))
  // output_strategy: ((n, o, h, w),)
  // dev_matrix: (n, i, h, w, o)
  TensorMap input_tensor_map = {4, 3, 2, 1};
  TensorMap weight_tensor_map = {0, 3, -1, -1};
  TensorMap output_tensor_map = {4, 0, 2, 1};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)inputs_tensor_map_.emplace_back(std::move(weight_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

// if in channel is split, it need to insert all reduce
Status Conv2DInfo::InferForwardCommunication() {
  forward_op_.clear();
  size_t relevant_dim_index = IN_CHANNEL_INDEX;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    // if repeated calculation and repeated num in the left of dev matrix, the index of relevant dimension should add 1
    relevant_dim_index += 1;
  }

  if (dev_matrix_shape_[relevant_dim_index] == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(relevant_dim_index, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed";
    return FAILED;
  }

  if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  Operator op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward all reduce is " << group_list[0].name();

  return SUCCESS;
}

ReplaceGraphPtr Conv2DInfo::replace_graph(const CNodePtr &cnode) {
  if (!out_channel_shard_) {
    return nullptr;
  }

  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  prim->set_attr(OUT_CHANNEL, MakeValue(new_out_channel_));
  return nullptr;
}

void Conv2DInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status Conv2DInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> Conv2DInfo::GenerateOpStrategies(int64_t stage_id) {
  Strategys strategy = {{stage_device_size_, 1, 1, 1}, {1, 1, 1, 1}};
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status Conv2DInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status Conv2DInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
