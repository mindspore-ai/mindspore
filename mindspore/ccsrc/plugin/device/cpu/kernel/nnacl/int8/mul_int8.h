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

#ifndef NNACL_INT8_MUL_INT8_H_
#define NNACL_INT8_MUL_INT8_H_

#include "nnacl/op_base.h"
#include "nnacl/mul_parameter.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/fixed_point.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
void Mul(const int8_t *input0_data, const int8_t *input1_data, int8_t *output_data, int64_t real_dst_count,
         const MulQuantArg *quant_arg);
void FastMul(const int8_t *input0_data, const int8_t *input1_data, int8_t *output_data, int depth,
             int64_t real_dst_count, bool input1_broad, const MulQuantArg *quant_arg);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_MUL_INT8_H_
