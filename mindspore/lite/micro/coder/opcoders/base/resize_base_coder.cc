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

#include "coder/opcoders/base/resize_base_coder.h"
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {
constexpr int kMaxInputNum = 2;
constexpr int kOutputNum = 1;
constexpr int kSingleNum = 1;
constexpr int kDoubleNum = 2;
constexpr int kQuadrupleNum = 4;

int ResizeBaseCoder::CheckParameters() {
  auto parameter = reinterpret_cast<ResizeParameter *>(parameter_);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "cast ResizeParameter failed.";
    return RET_NULL_PTR;
  }
  method_ = parameter->method_;
  if (method_ != static_cast<int>(schema::ResizeMethod_LINEAR) &&
      method_ != static_cast<int>(schema::ResizeMethod_NEAREST)) {
    MS_LOG(ERROR) << "Resize method should be bilinear or nearest_neighbor, but got " << method_;
    return RET_INVALID_OP_ATTR;
  }
  if (this->input_tensors_.size() == kSingleNum) {
    new_height_ = parameter->new_height_;
    if (new_height_ < 1) {
      MS_LOG(ERROR) << "Resize new_height should >= 1, but got " << new_height_;
      return RET_INVALID_OP_ATTR;
    }
    new_width_ = parameter->new_width_;
    if (new_width_ < 1) {
      MS_LOG(ERROR) << "Resize new_width should >= 1, but got " << new_width_;
      return RET_INVALID_OP_ATTR;
    }
  } else if (this->input_tensors_.size() == kDoubleNum) {
    auto out_shape = this->input_tensors_.at(1)->data_c();
    if (out_shape == nullptr) {
      MS_LOG(INFO) << "Out shape is not assigned";
      const_shape_ = false;
    } else {
      const_shape_ = true;
    }
  }
  coordinate_transform_mode_ = parameter->coordinate_transform_mode_;
  preserve_aspect_ratio_ = parameter->preserve_aspect_ratio_;
  if (preserve_aspect_ratio_) {
    MS_LOG(ERROR) << "Resize currently not support preserve_aspect_ratio true";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeBaseCoder::CheckInputsOuputs() {
  if (input_tensors_.size() <= kQuadrupleNum) {
    if (std::any_of(input_tensors_.begin(), input_tensors_.end(), [](const Tensor *t) { return t == nullptr; })) {
      return RET_NULL_PTR;
    }
  } else {
    MS_LOG(ERROR) << "Resize input num should be no more than" << kMaxInputNum << ", but got " << input_tensors_.size();
    return RET_ERROR;
  }
  if (output_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Resize output num should be " << kOutputNum << ", but got " << output_tensors_.size();
    return RET_ERROR;
  }
  auto output = output_tensors_.at(0);
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ResizeBaseCoder::Init() {
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckInputsOuputs();
  if (ret != RET_OK) {
    return ret;
  }
  auto input_shape = input_tensor_->shape();
  if (!input_shape.empty() && input_shape.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Resize op support input rank 4, got " << input_shape.size();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::micro
