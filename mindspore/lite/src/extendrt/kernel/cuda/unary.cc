/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/kernel/cuda/unary.h"
#include <memory>

namespace mindspore::kernel {
int UnaryCudaKernel::Prepare() {
  CudaKernel::Prepare();
  if (unary_helper_ == nullptr) {
    unary_helper_ = std::make_shared<cukernel::UnaryHelperGpuKernel<float>>(type_name_);
    helper_ = unary_helper_;
  }
  int ret = ReSize();
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}
int UnaryCudaKernel::Run() {
  int ret = unary_helper_->Process(input_device_ptrs_, output_device_ptrs_, work_device_ptrs_, stream_);
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}
// REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Log, CudaKernelCreator<UnaryCudaKernel>)
}  // namespace mindspore::kernel
