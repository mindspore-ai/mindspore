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

#include "src/litert/kernel/cpu/fp32/gather_d_fp32.h"
#include <limits>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GatherD;

namespace mindspore::kernel {
int GatherDCPUKernel::Run() { return RET_NOT_SUPPORT; }

int GatherDCPUKernel::AssignIndicesData(bool isIndicesInt32) { return RET_OK; }
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GatherD, LiteKernelCreator<GatherDCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GatherD, LiteKernelCreator<GatherDCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_GatherD, LiteKernelCreator<GatherDCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_GatherD, LiteKernelCreator<GatherDCPUKernel>)
}  // namespace mindspore::kernel
