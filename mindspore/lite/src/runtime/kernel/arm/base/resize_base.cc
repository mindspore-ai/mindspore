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
#include "src/runtime/kernel/arm/fp32/resize_fp32.h"
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kMaxInputNum = 2;
constexpr int kOutputNum = 1;
constexpr int kRank = 4;
}  // namespace

int ResizeBaseCPUKernel::CheckParameters() {
  auto parameter = reinterpret_cast<ResizeParameter *>(op_parameter_);
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
  if (this->in_tensors_.size() == lite::kSingleNum) {
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
  } else if (this->in_tensors_.size() == lite::kDoubleNum) {
    auto out_shape = this->in_tensors_.at(1)->data_c();
    if (out_shape == nullptr) {
      MS_LOG(INFO) << "Out shape is not assigned";
      const_shape_ = false;
    } else {
      auto ret = CalculateNewHeightWidth();
      if (ret != RET_OK) {
        return ret;
      }
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

int ResizeBaseCPUKernel::CalculateNewHeightWidth() {
  if (in_tensors_.size() != 2) {
    return RET_ERROR;
  }
  auto input_tensor = in_tensors_.at(0);
  auto shape_scale_tensor = in_tensors_.at(1);
  if (shape_scale_tensor->data_type() == kNumberTypeFloat32) {
    // float type means scale
    float *shape_scale = reinterpret_cast<float *>(shape_scale_tensor->data_c());
    if (shape_scale == nullptr) {
      return RET_ERROR;
    }
    if (shape_scale_tensor->format() == schema::Format_NHWC) {
      new_height_ = input_tensor->Height() * shape_scale[1];
      new_width_ = input_tensor->Width() * shape_scale[2];
    } else if (shape_scale_tensor->format() == schema::Format_NCHW) {
      new_height_ = input_tensor->Height() * shape_scale[2];
      new_width_ = input_tensor->Width() * shape_scale[3];
    } else {
      MS_LOG(ERROR) << "resize not support format " << shape_scale_tensor->format();
      return RET_ERROR;
    }
  } else if (shape_scale_tensor->data_type() == kNumberTypeInt32) {
    // int32 type means real shape
    int32_t *shape_data = reinterpret_cast<int32_t *>(shape_scale_tensor->data_c());
    if (shape_data == nullptr) {
      return RET_ERROR;
    }
    if (shape_scale_tensor->format() == schema::Format_NHWC) {
      new_height_ = shape_data[1];
      new_width_ = shape_data[2];
    } else if (shape_scale_tensor->format() == schema::Format_NCHW) {
      new_height_ = shape_data[2];
      new_width_ = shape_data[3];
    } else {
      MS_LOG(ERROR) << "resize not support format " << shape_scale_tensor->format();
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int ResizeBaseCPUKernel::CheckInputsOuputs() {
  if (in_tensors_.size() <= lite::kDoubleNum) {
    for (size_t i = 0; i < in_tensors_.size(); i++) {
      auto input = in_tensors_.at(i);
      if (input == nullptr) {
        return RET_NULL_PTR;
      }
    }
  } else {
    MS_LOG(ERROR) << "Resize input num should be no more than" << kMaxInputNum << ", but got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Resize output num should be " << kOutputNum << ", but got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
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
  if (!input_shape.empty() && input_shape.size() != kRank) {
    MS_LOG(ERROR) << "Resize op support input rank 4, got " << input_shape.size();
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
