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

#ifndef MINDSPORE_LITE_NNACL_INT8_TANH_INT8_H_
#define MINDSPORE_LITE_NNACL_INT8_TANH_INT8_H_

#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/int8/fixed_point.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/fp32/activation_fp32.h"

typedef struct TanhQuantParameter {
  int32_t in_zp_;
  int32_t out_zp_;
  double in_scale_;
  double out_scale_;
} TanhQuantParameter;

#ifdef __cplusplus
extern "C" {
#endif

void TanhInt8(const int8_t *input_ptr, int8_t *output_ptr, int size, TanhQuantParameter *quant);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_TANH_INT8_H_
