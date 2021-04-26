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
  return RET_OK;
}

int BroadcastToCPUKernel::Init() {
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
  const auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  return BroadcastTo(float, input_data, shape_info_, output_data);
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
}  // namespace mindspore::kernel
