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
#ifndef MINDSPORE_LITE_MICRO_CODER_WRAPPER_FP32_ARITHMETIC_FP32_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_WRAPPER_FP32_ARITHMETIC_FP32_WRAPPER_H_
#include "nnacl/fp32/arithmetic_fp32.h"
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum ArithmeticFuncType {
  kArithmeticFuncFloat = 0,
  kArithmeticFuncBool = 1,
  kArithmeticFuncInt = 2,
  kArithmeticFuncUnknow = 3,
} ArithmeticFuncType;

typedef struct ArithmeticWrapperInfo {
  int offset0_;
  int stride0_;
  int offset1_;
  int stride1_;
  int out_offset_;
  int out_stride_;
  ArithmeticFuncType arithmetic_func_type_;
} ArithmeticWrapperInfo;

typedef int (*ArithmeticRun)(const float *input0, const float *input1, float *output, const int element_size);
typedef int (*ArithmeticOptRun)(const float *input0, const float *input1, float *output, const int element_size,
                                const ArithmeticParameter *param);
typedef int (*ArithmeticIntRun)(const int *input0, const int *input1, int *output, const int element_size);
typedef int (*ArithmeticOptIntRun)(const int *input0, const int *input1, int *output, const int element_size,
                                   const ArithmeticParameter *param);
typedef int (*ArithmeticBoolRun)(const bool *input0, const bool *input1, bool *output, const int element_size);

void ArithmeticExecute(const void *input0, const void *input1, void *output, int size, bool is_opt,
                       ArithmeticFuncType arithmetic_func_type, const void *arithmetic_func,
                       const ArithmeticParameter *param);

void TileConstTensor(const float *in_data, float *out_data, size_t ndim, const int *in_shape, const int *in_strides,
                     const int *out_strides, const int *multiple);

void BatchScalarCalc(const void *input0, const void *input1, void *output, int batch_size, int size, bool is_opt,
                     const void *arithmetic_func, const ArithmeticWrapperInfo *wrapper_info,
                     const ArithmeticParameter *param);

void BroadcastRun(const void *input0, const void *input1, void *output, int dim, int out_count, int out_thread_stride,
                  int break_pos, int data_type_len, ArithmeticFuncType arithmetic_func_type,
                  const void *arithmetic_func, const ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CODER_WRAPPER_FP32_ARITHMETIC_FP32_WRAPPER_H_
