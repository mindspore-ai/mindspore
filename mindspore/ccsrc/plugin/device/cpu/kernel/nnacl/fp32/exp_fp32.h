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

#ifndef MINDSPORE_NNACL_FP32_EXP_H_
#define MINDSPORE_NNACL_FP32_EXP_H_

#include "nnacl/op_base.h"
#include "nnacl/kernel/exp.h"
#include "nnacl/exp_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void ExpFp32(const float *src, float *dst, int num);
int ExpFusionFp32(const void *src_data, void *dst_data, const ExpStruct *exp, int task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_EXP_H_
