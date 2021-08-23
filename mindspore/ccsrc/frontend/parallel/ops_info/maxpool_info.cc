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

#include "frontend/parallel/ops_info/maxpool_info.h"

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
Status MaxPoolInfo::GetAttrs() {
  // kernel_size
  kernel_size_ = GetTupleIntAttr(KERNEL_SIZE);
  if (kernel_size_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of kernel_size must be 4, but got " << kernel_size_.size();
    return FAILED;
  }

  if (kernel_size_[0] != 1 || kernel_size_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of kernel_size must be 1, but got (" << kernel_size_[0] << ", "
                  << kernel_size_[1] << ")";
    return FAILED;
  }

  // pad_mode
  pad_mode_ = GetIntAttr(PAD_MODE);
  if (pad_mode_ != POOL_PAD_MODE_VALID && pad_mode_ != POOL_PAD_MODE_SAME) {
    MS_LOG(ERROR) << name_ << ": The pad_mode value is invalid: " << pad_mode_;
    return FAILED;
  }

  // stride
  stride_ = GetTupleIntAttr(STRIDES);
  if (stride_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of stride must be 4, but got " << stride_.size();
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of stride must be 1, but got (" << stride_[0] << ", "
                  << stride_[1] << ")";
    return FAILED;
  }

  // format
  format_ = GetStringAttr(FORMAT);
  if (format_ != NCHW) {
    MS_LOG(ERROR) << name_ << ": The format only support 'NCHW', but got " << format_;
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The kernel size is " << kernel_size_ << ", pad mode is " << pad_mode_ << ", pad list is "
               << ", stride is " << stride_ << ", format is " << format_;

  return SUCCESS;
}

Status MaxPoolInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (outputs_shape_[0][2] % h_strategy != 0) {
    MS_LOG(ERROR) << name_
                  << ": Do not support to split h dimension when out_shape of h dimension is not divisible by strategy "
                     "of h dimension";
    return FAILED;
  }

  if (outputs_shape_[0][3] % w_strategy != 0) {
    MS_LOG(ERROR) << name_
                  << ": Do not support to split w dimension when out_shape of w dimension is not divisible by strategy "
                     "of w dimension";
    return FAILED;
  }

  if (h_strategy > 1) {
    if (kernel_size_[2] > stride_[2]) {
      MS_LOG(ERROR) << name_ << ": It does not support to split H dimension when kernel_size > stride";
      return FAILED;
    }

    int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
    if (h_slice_shape % stride_[2] != 0) {
      MS_LOG(ERROR) << name_
                    << ": It does not support to split H dimension when kernel_size <= stride but slice shape is not "
                       "divisible by stride ";
      return FAILED;
    }
  }

  if (w_strategy > 1) {
    if (kernel_size_[3] > stride_[3]) {
      MS_LOG(ERROR) << name_ << ": It does not support to split W dimension when kernel_size > stride";
      return FAILED;
    }

    int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;
    if (w_slice_shape % stride_[3] != 0) {
      MS_LOG(ERROR) << name_
                    << ": It does not support to split W dimension when kernel_size <= stride but slice shape is not "
                       "divisible by stride ";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MaxPoolInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 1, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  if (input_strategy.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of input strategy must be 4, but got" << input_strategy.size();
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MaxPoolInfo::InferDevMatrixShape() {
  // the strategy is (n, c, h, w)
  // the dev matrix is (n, c, h, w)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status MaxPoolInfo::InferTensorMap() {
  // input_strategy: (n, c, h, w)
  // output_strategy: (n, c, h, w)
  // dev_matrix: (n, c, h, w)
  TensorMap input_tensor_map = {3, 2, 1, 0};
  TensorMap output_tensor_map = {3, 2, 1, 0};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status MaxPoolInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> MaxPoolInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 0);
  input0_split[0] = 1;
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

Status MaxPoolInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status MaxPoolInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
