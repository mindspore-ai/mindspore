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

#include "src/extendrt/kernel/cuda/unique.h"
#include <memory>
#include <vector>

namespace mindspore::kernel {
int UniqueCudaKernel::Prepare() {
  CudaKernel::Prepare();
  if (unique_helper_ == nullptr) {
    unique_helper_ = std::make_shared<cukernel::UniqueHelperGpuKernel<int, int>>(type_name_);
    helper_ = unique_helper_;
  }
  return RET_OK;
}

int UniqueCudaKernel::PostProcess() {
  auto ret = CudaKernel::PostProcess();
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  // set output tensor shape
  std::vector<int> out_shape = out_tensors_[0]->shape();
  out_shape[out_shape.size() - 1] = unique_helper_->GetOutSize();
  out_tensors_[0]->set_shape(out_shape);
  return RET_OK;
}

int UniqueCudaKernel::Run() {
  int ret = unique_helper_->Process(input_device_ptrs_, output_device_ptrs_, work_device_ptrs_, stream_);
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}
// REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Unique, CudaKernelCreator<UniqueCudaKernel>)
}  // namespace mindspore::kernel
