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
#ifndef MINDSPORE_NNACL_FP16_ACTIVATION_FP16_H_
#define MINDSPORE_NNACL_FP16_ACTIVATION_FP16_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/int8/fixed_point.h"
#include "nnacl/activation_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int ReluFp16(const float16_t *src, float16_t *dst, int ele_num);
int Relu6Fp16(const float16_t *data, float16_t *dst, int ele_num);
int LReluFp16(const float16_t *src, float16_t *dst, int ele_num, float16_t alpha);
int SigmoidFp16(const float16_t *src, float16_t *dst, int ele_num);
int TanhFp16(const float16_t *src, float16_t *dst, int ele_num);
int HSwishFp16(const float16_t *src, float16_t *dst, int ele_num);
int HSigmoidFp16(const float16_t *src, float16_t *dst, int ele_num);
int SwishFp16(const float16_t *src, float16_t *dst, int ele_num);
int HardTanhFp16(const float16_t *src, int length, float16_t *dst, float min_val, float max_val);
int GeluFp16(const float16_t *src, int length, float16_t *dst, bool approximate);
int SoftplusFp16(const float16_t *src, int length, float16_t *dst);
int EluFp16(const float16_t *src, int length, float16_t *dst, float16_t alpha);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP16_ACTIVATION_FP16_H_
