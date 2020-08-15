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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_WINOGRAD_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_WINOGRAD_UTILS_H_

#include <arm_neon.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

typedef void (*InputTransformUnitFp16Func)(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step);
typedef void (*OutputTransformUnitFp16Func)(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                            int src_step, int dst_step);

#ifdef __cplusplus
extern "C" {
#endif
void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step);

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step);

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step);

InputTransformUnitFp16Func GetInputTransFuncFp16(int input_unit);

OutputTransformUnitFp16Func GetOutputTransFuncFp16(int input_unit, int output_unit);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_WINOGRAD_UTILS_H_
