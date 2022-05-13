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

#include "src/extendrt/kernel/cuda/batchtospace.h"
#include <memory>
#include "nnacl/batch_to_space.h"

namespace mindspore::kernel {
int BatchtoSpaceCudaKernel::Prepare() {
  CudaKernel::Prepare();
  if (batch_to_space_helper_ == nullptr) {
    batch_to_space_helper_ = std::make_shared<cukernel::BatchToSpaceHelperGpuKernel<float>>(type_name_);
    helper_ = batch_to_space_helper_;
  }
  cukernel::BatchToSpaceAttr attr;
  auto param = reinterpret_cast<BatchToSpaceParameter *>(op_parameter_);
  attr.block_size = param->block_shape_[0];
  attr.crops.push_back({param->crops_[0], param->crops_[1]});
  attr.crops.push_back({param->crops_[2], param->crops_[3]});
  int ret = batch_to_space_helper_->CheckKernelParam(&attr);
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  ret = ReSize();
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}

int BatchtoSpaceCudaKernel::Run() {
  int ret = batch_to_space_helper_->Process(input_device_ptrs_, output_device_ptrs_, work_device_ptrs_, stream_);
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}

// REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, CudaKernelCreator<BatchtoSpaceCudaKernel>)
}  // namespace mindspore::kernel
