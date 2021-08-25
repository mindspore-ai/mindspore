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
#include "src/runtime/kernel/arm/fp32/broadcast_to_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BroadcastTo;

namespace mindspore::kernel {
constexpr int kBroadCastToInputNum = 2;

BroadcastToCPUKernel::~BroadcastToCPUKernel() {
  if (shape_info_ != nullptr) {
    free(shape_info_);
    shape_info_ = nullptr;
  }
}

int BroadcastToCPUKernel::ReSize() {
  auto input_shape = in_tensors_.at(0)->shape();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    shape_info_->input_shape_[i] = input_shape[i];
  }

  shape_info_->input_shape_size_ = static_cast<int>(input_shape.size());
  auto output_shape = out_tensors_.at(0)->shape();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    shape_info_->output_shape_[i] = output_shape[i];
  }
  shape_info_->output_shape_size_ = static_cast<int>(output_shape.size());

  data_type_ = in_tensors_.at(0)->data_type();
  MS_ASSERT(data_type_ == out_tensors_.at(0)->data_type());
  return RET_OK;
}

int BroadcastToCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1)
  CHECK_LESS_RETURN(out_tensors_.size(), 1)
  shape_info_ = reinterpret_cast<BroadcastShapeInfo *>(malloc(sizeof(BroadcastShapeInfo)));
  if (shape_info_ == nullptr) {
    MS_LOG(ERROR) << "Malloc BroadcastShapeInfo failed!";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BroadcastToCPUKernel::Run() {
  if (in_tensors_.size() == kBroadCastToInputNum) {
    auto shape_tensor = in_tensors_.at(1);
    MS_ASSERT(shape_tensor->data_type() == kNumberTypeInt32);
    if (shape_tensor->ElementsNum() > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "Size of broadcast_to shape exceed MAX_SHAPE_SIZE";
      return RET_ERROR;
    }
    auto shape_data = reinterpret_cast<int *>(shape_tensor->data());
    for (int i = 0; i < shape_tensor->ElementsNum(); i++) {
      shape_info_->output_shape_[i] = (shape_data[i] == -1) ? (shape_info_->input_shape_[i]) : shape_data[i];
    }
  }
  switch (data_type_) {
    case kNumberTypeFloat32: {
      const auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
      auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
      return BroadcastTo(float, input_data, shape_info_, output_data);
    }
    case kNumberTypeInt32:
    case kNumberTypeInt: {
      const auto input_data = reinterpret_cast<int *>(in_tensors_.at(0)->data_c());
      auto output_data = reinterpret_cast<int *>(out_tensors_.at(0)->data_c());
      return BroadcastTo(int, input_data, shape_info_, output_data);
    }
    default:
      MS_LOG(ERROR) << "UnSupported data type: " << data_type_;
      return RET_ERROR;
  }
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
}  // namespace mindspore::kernel
