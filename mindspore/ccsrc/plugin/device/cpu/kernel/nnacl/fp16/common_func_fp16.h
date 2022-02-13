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
#ifndef MINDSPORE_NNACL_FP16_COMMON_FUNC_FP16_H_
#define MINDSPORE_NNACL_FP16_COMMON_FUNC_FP16_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* deconv common */
void PostConvFuncFp16C8(const float16_t *c8_out_ptr, float16_t *out_ptr, const float16_t *bias_ptr,
                        size_t output_channel, size_t plane_size, size_t stride, ActType act_type);
void PostFuncBiasReluC8Fp16(float16_t *dst, const float16_t *src, const float16_t *bias, size_t oc8div, size_t oc8mod,
                            size_t plane_size, size_t stride, size_t relu_type);

/* deconv winograd */
void PostConvFuncFp16C4(const float16_t *c4_out, float16_t *nhwc_out, const float16_t *bias, size_t output_channel,
                        size_t plane_size, size_t plane_stride, ActType act_type);
void PostFuncBiasReluC4Fp16(float16_t *dst, const float16_t *src, const float16_t *bias, size_t oc4div, size_t oc4mod,
                            size_t plane_size, size_t plane_stride, size_t relu_type);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP16_COMMON_FUNC_FP16_H_
