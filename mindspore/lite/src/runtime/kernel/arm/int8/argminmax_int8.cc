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
#include "src/runtime/kernel/arm/int8/argminmax_int8.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/arg_min_max.h"
#include "include/errorcode.h"

using mindspore::lite::RET_OK;
using mindspore::lite::RET_ERROR;

namespace mindspore::kernel {
int ArgMinMaxInt8CPUKernel::Init() {
  auto ret = ArgMinMaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto param = reinterpret_cast<ArgMinMaxParameter *>(opParameter);
  param->data_type_ = kNumberTypeInt8;
  return RET_OK;
}

int ArgMinMaxInt8CPUKernel::Run() {
  auto ret = ArgMinMaxBaseCPUKernel::Run();
  FreeTmpMemory();
  return ret;
}
}  // namespace mindspore::kernel
