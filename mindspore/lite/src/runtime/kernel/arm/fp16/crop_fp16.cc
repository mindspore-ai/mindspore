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
#include "src/runtime/kernel/arm/fp16/crop_fp16.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
int CropFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CropFp16CPUKernel::DoExecute(int task_id) {
  Fp16Crop(input_ptr_, output_ptr_, task_id, crop_para_);
  return RET_OK;
}

static int CropFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<CropFp16CPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CropRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(output_tensor != nullptr);
  input_ptr_ = reinterpret_cast<float16_t *>(input_tensor->data());
  output_ptr_ = reinterpret_cast<float16_t *>(output_tensor->data());
  MS_ASSERT(input_ptr_ != nullptr);
  MS_ASSERT(output_ptr_ != nullptr);
  auto ret = ParallelLaunch(this->ms_context_, CropFp16Run, this, crop_para_->thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Crop, LiteKernelCreator<CropFp16CPUKernel>)
}  // namespace mindspore::kernel
