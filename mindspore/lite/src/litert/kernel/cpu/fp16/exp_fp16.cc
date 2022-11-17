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
#include "src/litert/kernel/cpu/fp16/exp_fp16.h"
#include "nnacl/fp16/exp_fp16.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore::kernel {
int ExpFp16CPUKernel::DoExcute(int task_id) {
  CHECK_NULL_RETURN(input_addr_);
  CHECK_NULL_RETURN(output_addr_);
  ExpFusionFp16(reinterpret_cast<float16_t *>(input_addr_), reinterpret_cast<float16_t *>(output_addr_), param_,
                task_id);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ExpFusion, LiteKernelCreator<ExpFp16CPUKernel>)
}  // namespace mindspore::kernel
