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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_STRASSEN_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_STRASSEN_MATMUL_H_

#include <memory.h>
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/strassen_matmul.h"
#include "nnacl/fp32/common_func.h"

#define FP32_STRASSEN_UINT C4NUM
#define FP32_STRASSEN_WEIGHT_UINT (C4NUM * C4NUM)
#define FP32_STRASSEN_MAX_RECURSION 5

#ifdef __cplusplus
extern "C" {
#endif
int RecursionMatmul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *matmul_param,
                    int max_recursion, int, float *tmp_a_ptr);
int CommonMatMul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *Matmul_param,
                 float *tmp_a_ptr);

int StrassenMatmul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *matmul_param,
                   int max_recursion, int cur_recursion, float *tmp_a_ptr);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_STRASSEN_MATMUL_H_
