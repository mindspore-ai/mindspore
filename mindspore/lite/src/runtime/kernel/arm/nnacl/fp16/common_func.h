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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_COMMON_FUNC_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_COMMON_FUNC_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ENABLE_ARM64
void ConvDwFp16Border(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                      size_t height, size_t width, size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu,
                      size_t relu6);
void ConvDwFp16Center(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                      size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t out_h_step,
                      size_t block_channel, size_t in_sh_step, size_t in_sw_step, size_t in_kh_step, size_t in_kw_step,
                      size_t relu, size_t relu6);
void DeconvDwFp16Center(float16_t *dst, const float16_t *src, const float16_t *weight, size_t height, size_t width,
                        size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                        size_t in_sw_step, size_t in_kh_step, size_t in_kw_step);
#endif
void ReluFp16(float16_t *data, float16_t *dst, int ele_num);
void Relu6Fp16(float16_t *data, float16_t *dst, int ele_num);

#ifdef __cplusplus
}
#endif

#endif /* MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_COMMON_FUNC_H_ */
