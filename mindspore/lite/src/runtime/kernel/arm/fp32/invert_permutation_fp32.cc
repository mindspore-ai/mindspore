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

#include "src/runtime/kernel/arm/fp32/invert_permutation_fp32.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_InvertPermutation;

namespace mindspore::kernel {
int InvertPermutationCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int InvertPermutationCPUKernel::ReSize() {
  if (in_tensors_[0]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "InvertPermutation does not support input of data type: " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() != 1) {
    MS_LOG(ERROR) << "InvertPermutation input must be one-dimensional.";
    return RET_ERROR;
  }
  return RET_OK;
}

int InvertPermutationCPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }
  auto input_ptr = reinterpret_cast<int32_t *>(in_tensor->data_c());
  auto output_ptr = reinterpret_cast<int32_t *>(out_tensor->data_c());
  if (input_ptr == nullptr || output_ptr == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }
  InvertPermutation(input_ptr, output_ptr, in_tensors_[0]->ElementsNum());
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_InvertPermutation, LiteKernelCreator<InvertPermutationCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_InvertPermutation, LiteKernelCreator<InvertPermutationCPUKernel>)
}  // namespace mindspore::kernel
