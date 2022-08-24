/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/cpu/fp16/scale_fp16.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/runtime/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/cpu/fp16/common_fp16.h"
#include "nnacl/fp16/scale_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
void ScaleFp16CPUKernel::FreeRunningBuffer() {
  for (auto buffer : run_buffers_) {
    ms_context_->allocator->Free(buffer);
  }
  run_buffers_.clear();
}

int ScaleFp16CPUKernel::EnsureFp16Inputs() {
  if (in_tensors_[kWeightIndex]->data_type() == kNumberTypeFloat32 ||
      in_tensors_[kWeightIndex]->data_type() == kNumberTypeFloat) {
    scale_ =
      ConvertInputFp32toFp16(in_tensors_[kWeightIndex], static_cast<const lite::InnerContext *>(this->ms_context_));
    if (scale_ == nullptr) {
      MS_LOG(ERROR) << "ScaleFp16: convert second-input from fp32 to fp16 failed.";
      return RET_NULL_PTR;
    }
    run_buffers_.push_back(scale_);
  } else {
    scale_ = in_tensors_[kWeightIndex]->data();
  }
  if (in_tensors_.size() == kInputSize1) {
    return RET_OK;
  }
  if (in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat32 ||
      in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat) {
    offset_ =
      ConvertInputFp32toFp16(in_tensors_[kBiasIndex], static_cast<const lite::InnerContext *>(this->ms_context_));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "ScaleFp16: convert third-input from fp32 to fp16 failed.";
      return RET_NULL_PTR;
    }
    run_buffers_.push_back(offset_);
  } else {
    offset_ = in_tensors_[kBiasIndex]->data();
  }
  return RET_OK;
}

int ScaleFp16CPUKernel::Compute(int task_id) {
  if (task_id + 1 >= static_cast<int>(split_points_.size())) {
    return RET_OK;
  }
  int block[C2NUM] = {static_cast<int>(split_points_[task_id]), static_cast<int>(split_points_[task_id + 1])};
  DoScaleFp16(static_cast<float16_t *>(input_ptr_), static_cast<float16_t *>(scale_), static_cast<float16_t *>(offset_),
              static_cast<float16_t *>(output_ptr_), scale_param_, block);
  return RET_OK;
}

int ScaleFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_[kInputIndex];
  auto output_tensor = out_tensors_[kInputIndex];
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(output_tensor);
  input_ptr_ = input_tensor->data();
  output_ptr_ = output_tensor->data();
  auto ret = EnsureFp16Inputs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleFp16: do EnsureFp16Inputs failed.";
    FreeRunningBuffer();
    return ret;
  }
  ret = ScaleBaseCPUKernel::Run();
  FreeRunningBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleFp16: running failed.";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ScaleFusion, LiteKernelCreator<ScaleFp16CPUKernel>)
}  // namespace mindspore::kernel
