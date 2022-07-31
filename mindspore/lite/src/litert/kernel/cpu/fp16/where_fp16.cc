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
#include "src/litert/kernel/cpu/fp16/where_fp16.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "nnacl/fp16/where_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Where;

namespace mindspore::kernel {
int WhereFp16CPUKernel::DoExcute(int task_id) {
  CHECK_NULL_RETURN(condition_);
  CHECK_NULL_RETURN(x_);
  CHECK_NULL_RETURN(y_);
  CHECK_NULL_RETURN(output_data_);
  CHECK_NULL_RETURN(where_param_);
  WhereWithTripleInputsFp16(condition_, static_cast<float16_t *>(x_), static_cast<float16_t *>(y_),
                            static_cast<float16_t *>(output_data_), where_param_, task_id);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Where, LiteKernelCreator<WhereFp16CPUKernel>)
}  // namespace mindspore::kernel
