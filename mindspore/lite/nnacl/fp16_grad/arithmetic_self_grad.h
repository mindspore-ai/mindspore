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
#ifndef MINDSPORE_LITE_NNACL_FP16_GRAD_ARITHMETHIC_SELF_GRAD_H_
#define MINDSPORE_LITE_NNACL_FP16_GRAD_ARITHMETHIC_SELF_GRAD_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <math.h>
#include "nnacl/op_base.h"

typedef struct ArithmeticSelfGradParameterFp16 {
  OpParameter op_parameter;
  int type_;
} ArithmeticSelfGradParameterFp16;
#ifdef __cplusplus
extern "C" {
#endif

int Fp16LogGrad(const float16_t *src0, const float16_t *src1, size_t length, float16_t *dst);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_GRAD_ARITHMETHIC_SELF_GRAD_H_
