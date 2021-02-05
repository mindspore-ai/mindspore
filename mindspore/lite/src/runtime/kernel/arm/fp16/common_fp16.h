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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_COMMON_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_COMMON_FP16_H_

#include <vector>
#include "src/lite_kernel.h"

namespace mindspore::kernel {
float16_t *ConvertInputFp32toFp16(lite::Tensor *input, const lite::InnerContext *ctx);

float16_t *MallocOutputFp16(lite::Tensor *output, const lite::InnerContext *ctx);

int ConvertFp32TensorToFp16(lite::Tensor *tensor, const lite::InnerContext *ctx);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_COMMON_FP16_H_
