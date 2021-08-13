/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/base/partial_fusion.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif
#include "src/common/utils.h"

// this file is going to be removed when move create actor before schedule.
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PartialFusion;

namespace mindspore::kernel {
int PartialFusionKernel::Init() { return RET_OK; }
int PartialFusionKernel::ReSize() { return RET_OK; }
int PartialFusionKernel::Run() { return RET_OK; }
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PartialFusion, LiteKernelCreator<PartialFusionKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_PartialFusion, LiteKernelCreator<PartialFusionKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_PartialFusion, LiteKernelCreator<PartialFusionKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_PartialFusion, LiteKernelCreator<PartialFusionKernel>)
}  // namespace mindspore::kernel
