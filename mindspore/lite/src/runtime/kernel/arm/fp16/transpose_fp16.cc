/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp16/transpose_fp16.h"
#include <vector>
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/transpose_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
void TransposeFp16CPUKernel::GetNchwToNhwcFunc(TypeId dtype) { NHNCTransposeFunc_ = PackNCHWToNHWCFp16; }

void TransposeFp16CPUKernel::GetNhwcToNchwFunc(TypeId dtype) { NHNCTransposeFunc_ = PackNHWCToNCHWFp16; }

int TransposeFp16CPUKernel::TransposeDim2to6() {
  return DoTransposeFp16(static_cast<const float16_t *>(in_data_), static_cast<float16_t *>(out_data_), out_shape_,
                         param_);
}

int TransposeFp16CPUKernel::TransposeDimGreaterThan6(int task_id) {
  TransposeDimsFp16(static_cast<const float16_t *>(in_data_), static_cast<float16_t *>(out_data_), out_shape_, param_,
                    task_id, op_parameter_->thread_num_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Transpose, LiteKernelCreator<TransposeFp16CPUKernel>)
}  // namespace mindspore::kernel
