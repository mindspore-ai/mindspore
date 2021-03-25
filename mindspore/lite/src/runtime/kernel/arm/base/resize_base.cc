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

#include <vector>
#include "src/runtime/kernel/arm/base/resize_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kMaxInputNum = 4;
constexpr int kOutputNum = 1;
}  // namespace

int ResizeBaseCPUKernel::CheckParameters() {
  auto parameter = reinterpret_cast<ResizeParameter *>(op_parameter_);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "cast ResizeParameter failed.";
    return RET_NULL_PTR;
  }
  method_ = parameter->method_;
  if (method_ == schema::ResizeMethod::ResizeMethod_UNKNOWN) {
    MS_LOG(ERROR) << "Resize method can not be unknown.";
    return RET_INVALID_OP_ATTR;
  }
  if (this->in_tensors_.size() == 1) {
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
  } else if (this->in_tensors_.size() == 2) {
    auto out_shape = this->in_tensors_.at(1)->data_c();
    if (out_shape == nullptr) {
      MS_LOG(INFO) << "Out shape is not assigned";
      const_shape_ = false;
    } else {
      if (InferShapeDone()) {
        new_height_ = out_tensors_.at(0)->shape().at(1);
        new_width_ = out_tensors_.at(0)->shape().at(2);
        const_shape_ = true;
      }
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

int ResizeBaseCPUKernel::CheckInputsOuputs() {
  // inputs
  if (in_tensors_.size() <= kMaxInputNum) {
    for (auto input : in_tensors_) {
      MSLITE_CHECK_PTR(input);
    }
  } else {
    MS_LOG(ERROR) << "Resize input num should be no more than" << kMaxInputNum << ", but got " << in_tensors_.size();
    return RET_ERROR;
  }

  // outputs
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Resize output num should be " << kOutputNum << ", but got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  MSLITE_CHECK_PTR(output);
  return RET_OK;
}

int ResizeBaseCPUKernel::Init() {
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckInputsOuputs();
  if (ret != RET_OK) {
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto input_shape = input->shape();
  if (!input_shape.empty() && input_shape.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Resize op support input rank 4, got " << input_shape.size();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
