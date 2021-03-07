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

#include "coder/opcoders/base/reduce_base_coder.h"
#include <vector>
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace
int ReduceBaseCoder::CheckInputsOutputs() const {
  if (input_tensors_.size() < kInputNum) {
    MS_LOG(ERROR) << "Reduce inputs size should be at least " << kInputNum << " but got " << input_tensors_.size();
    return RET_ERROR;
  }
  if (output_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Reduce outputs size should be " << kOutputNum << " but got " << output_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceBaseCoder::CheckParameters() {
  size_t input_rank = input_tensor_->shape().size();
  if (static_cast<size_t>(num_axes_) > input_rank) {
    MS_LOG(ERROR) << "Reduce op invalid num of reduce axes " << num_axes_ << " larger than input rank " << input_rank;
    return RET_ERROR;
  }

  for (auto i = 0; i < num_axes_; i++) {
    if (axes_[i] < -static_cast<int>(input_rank) || axes_[i] >= static_cast<int>(input_rank)) {
      MS_LOG(ERROR) << "Reduce got invalid axis " << axes_[i] << ", axis should be in ["
                    << -static_cast<int>(input_rank) << ", " << input_rank - 1 << "].";
      return RET_ERROR;
    }
    if (axes_[i] < 0) {
      axes_[i] += static_cast<int>(input_rank);
    }
  }

  if (reduce_to_end_) {
    // actual num of axes to reduce
    num_axes_ = static_cast<int>(input_rank) - axes_[0];
    MS_CHECK_TRUE(num_axes_ <= MAX_SHAPE_SIZE, "invalid num_axes_, greater than 8.");
    for (auto i = 1; i < num_axes_; ++i) {
      axes_[i] = axes_[0] + i;
    }
  }

  if (num_axes_ == 0) {
    for (size_t i = 0; i < input_rank; i++) {
      axes_[i] = i;
    }
    num_axes_ = static_cast<int>(input_rank);
  }

  return RET_OK;
}

int ReduceBaseCoder::Init() {
  auto reduce_param = reinterpret_cast<ReduceParameter *>(parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }
  if (input_tensors_.size() > 1) {
    Tensor *axes_ptr = input_tensors_.at(1);
    num_axes_ = axes_ptr->ElementsNum();
    MS_CHECK_PTR(axes_ptr->MutableData());
    MS_CHECK_RET_CODE(memcpy_s(axes_, sizeof(axes_), axes_ptr->MutableData(), axes_ptr->Size()), "memcpy_s failed");
  } else {
    num_axes_ = reduce_param->num_axes_;
    MS_CHECK_RET_CODE(memcpy_s(axes_, sizeof(axes_), reduce_param->axes_, sizeof(reduce_param->axes_)),
                      "memcpy_s failed!");
  }
  mode_ = reduce_param->mode_;
  MS_CHECK_RET_CODE(memcpy_s(axes_, sizeof(axes_), reduce_param->axes_, sizeof(reduce_param->axes_)),
                    "memcpy_s failed!");
  reduce_to_end_ = reduce_param->reduce_to_end_;
  MS_CHECK_RET_CODE(CheckInputsOutputs(), "CheckInputsOutputs failed!");
  return RET_OK;
}

void ReduceBaseCoder::CalculateInnerOuterSize() {
  outer_sizes_.clear();
  inner_sizes_.clear();
  axis_sizes_.clear();
  std::vector<int> tmp_shape = input_tensors_.at(0)->shape();
  for (int i = 0; i < num_axes_; ++i) {
    int axis = axes_[i];
    int outer_size = 1;
    for (int j = 0; j < axis; j++) {
      outer_size *= tmp_shape.at(j);
    }
    outer_sizes_.emplace_back(outer_size);
    int inner_size = 1;
    for (int k = axis + 1; k < static_cast<int>(tmp_shape.size()); k++) {
      inner_size *= tmp_shape.at(k);
    }
    inner_sizes_.emplace_back(inner_size);
    axis_sizes_.emplace_back(tmp_shape[axis]);
    tmp_shape[axis] = 1;
  }
}

void ReduceBaseCoder::CalculateTmpBufferSize() {
  buffer_sizes_.clear();
  std::vector<int> input_shape = input_tensor_->shape();
  for (int i = 0; i < num_axes_; i++) {
    int axis = axes_[i];
    size_t size = 1;
    for (int j = 0; j < static_cast<int>(input_shape.size()); j++) {
      if (axis != j) {
        size *= input_shape.at(j);
      }
    }
    buffer_sizes_.emplace_back(size);
    input_shape[axis] = 1;
  }
}

int ReduceBaseCoder::ReSize() {
  int ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  CalculateTmpBufferSize();
  CalculateInnerOuterSize();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
