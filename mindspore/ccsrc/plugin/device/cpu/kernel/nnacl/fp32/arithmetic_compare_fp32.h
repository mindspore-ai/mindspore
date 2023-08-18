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

#ifndef MINDSPORE_NNACL_ARITHMETIC_COMPARE_H_
#define MINDSPORE_NNACL_ARITHMETIC_COMPARE_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementEqualBool(const bool *input0, const bool *input1, uint8_t *output, int element_size);
int ElementOptEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size, bool first_scalar);
int ElementEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                         bool first_scalar);

int ElementNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementOptNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                           bool first_scalar);
int ElementNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                            bool first_scalar);
int ElementNotEqualInt64(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size);
int ElementOptNotEqualInt64(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size,
                            bool first_scalar);

int ElementLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementOptLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size, bool first_scalar);
int ElementLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                        bool first_scalar);

int ElementLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementOptLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                            bool first_scalar);
int ElementLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                             bool first_scalar);

int ElementGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementOptGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                          bool first_scalar);
int ElementGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                           bool first_scalar);

int ElementGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementOptGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                               bool first_scalar);
int ElementGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
int ElementOptGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                                bool first_scalar);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_ARITHMETIC_COMPARE_H_
