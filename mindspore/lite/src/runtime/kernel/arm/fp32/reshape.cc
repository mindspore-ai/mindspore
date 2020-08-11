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

#include "src/runtime/kernel/arm/fp32/reshape.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/reshape.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::kernel {
int ReshapeCPUKernel::Init() {
  ReshapeBaseCPUKernel::Init();
  return RET_OK;
}

int ReshapeCPUKernel::ReSize() { return RET_OK; }

int ReshapeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_ptr = in_tensors_.at(kInputIndex)->Data();
  auto output_ptr = out_tensors_.at(kOutputIndex)->Data();
  size_t data_size = in_tensors_.at(kInputIndex)->Size();
  Reshape(input_ptr, output_ptr, data_size);
  return RET_OK;
}
}  // namespace mindspore::kernel
