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

#include "src/runtime/kernel/arm/fp32/zeroslike_fp32.h"
#include "schema/model_generated.h"
#include "mindspore/lite/nnacl/base/zeroslike_base.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ZerosLike;

namespace mindspore::kernel {
int ZerosLikeCPUKernel::Init() { return RET_OK; }

int ZerosLikeCPUKernel::Run() {
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  ApproximateZerosLike(output_data, in_tensors_.at(0)->ElementsNum(), sizeof(float));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ZerosLike, LiteKernelCreator<ZerosLikeCPUKernel>)
}  // namespace mindspore::kernel
