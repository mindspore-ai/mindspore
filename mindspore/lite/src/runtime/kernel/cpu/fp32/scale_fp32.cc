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

#include "src/runtime/kernel/cpu/fp32/scale_fp32.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
int ScaleCPUKernel::Compute(int task_id) {
  if (task_id + 1 >= static_cast<int>(split_points_.size())) {
    return RET_OK;
  }
  int block[C2NUM] = {static_cast<int>(split_points_[task_id]), static_cast<int>(split_points_[task_id + 1])};
  DoScaleFp32(static_cast<float *>(input_ptr_), static_cast<float *>(scale_), static_cast<float *>(offset_),
              static_cast<float *>(output_ptr_), scale_param_, block);
  return RET_OK;
}

int ScaleCPUKernel::Run() {
  input_ptr_ = reinterpret_cast<float *>(in_tensors_[kInputIndex]->data());
  CHECK_NULL_RETURN(input_ptr_);
  scale_ = reinterpret_cast<float *>(in_tensors_[kWeightIndex]->data());
  CHECK_NULL_RETURN(scale_);
  if (in_tensors_.size() == kInputSize2) {
    offset_ = reinterpret_cast<float *>(in_tensors_[kBiasIndex]->data());
  }
  output_ptr_ = reinterpret_cast<float *>(out_tensors_[kInputIndex]->data());
  return ScaleBaseCPUKernel::Run();
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScaleFusion, LiteKernelCreator<ScaleCPUKernel>)
}  // namespace mindspore::kernel
