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

#include "src/runtime/kernel/arm/fp32/squeeze_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {
int SqueezeCPUKernel::Init() { return RET_OK; }

int SqueezeCPUKernel::ReSize() { return RET_OK; }

int SqueezeCPUKernel::Run() {
  size_t data_size = in_tensors_.front()->Size();
  int ret = DoSqueeze(in_tensors_.front()->data_c(), out_tensors_.front()->data_c(), data_size);

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do squeeze fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Squeeze, LiteKernelCreator<SqueezeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Squeeze, LiteKernelCreator<SqueezeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Squeeze, LiteKernelCreator<SqueezeCPUKernel>)
}  // namespace mindspore::kernel
