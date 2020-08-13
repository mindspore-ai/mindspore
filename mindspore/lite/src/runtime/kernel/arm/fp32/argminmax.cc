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

#include "src/runtime/kernel/arm/fp32/argminmax.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/arg_min_max.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMax;
using mindspore::schema::PrimitiveType_ArgMin;

namespace mindspore::kernel {
int ArgMinMaxCPUKernel::Init() {
  auto ret = ArgMinMaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  param->data_type_ = kNumberTypeFloat32;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxCPUKernel::ReSize() {
  ArgMinMaxBaseCPUKernel::FreeTmpMemory();
  return ArgMinMaxBaseCPUKernel::ReSize();
}

int ArgMinMaxCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  ret = ArgMinMaxBaseCPUKernel::Run();
  return ret;
}
}  // namespace mindspore::kernel
