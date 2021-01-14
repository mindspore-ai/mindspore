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
#include "src/runtime/kernel/arm/fp16/slice_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/kernel_registry.h"
#include "nnacl/fp16/slice_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;

namespace mindspore::kernel {
int SliceFp16CPUKernel::SliceParallelRun(int thread_id) {
  DoSliceFp16(input_fp16_, output_fp16_, param_, thread_id);
  return RET_OK;
}

int SliceFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);

  input_fp16_ = reinterpret_cast<float16_t *>(input_tensor->data_c());
  output_fp16_ = reinterpret_cast<float16_t *>(output_tensor->data_c());

  if (param_->size_[1] < op_parameter_->thread_num_) {
    DoSliceFp16NoParallel(input_fp16_, output_fp16_, param_);
    return RET_OK;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, SliceLaunch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "slice launch fail!ret: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Slice, LiteKernelCreator<SliceFp16CPUKernel>)
}  // namespace mindspore::kernel
