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

#ifndef MINDSPORE_NNACL_FP16_DECONV_FP16_H_
#define MINDSPORE_NNACL_FP16_DECONV_FP16_H_

#include <arm_neon.h>
#include <string.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp16/common_func_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

int DeConvPostFp16(const float16_t *src, float16_t *tmp, const float16_t *bias, float16_t *dst, int output_channel,
                   const ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP16_DECONV_FP16_H_
