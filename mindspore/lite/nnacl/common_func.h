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

#ifndef MINDSPORE_LITE_NNACL_COMMON_FUNC_H_
#define MINDSPORE_LITE_NNACL_COMMON_FUNC_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int8_t MinInt8(int8_t a, int8_t b);
int8_t MaxInt8(int8_t a, int8_t b);
void ReluFp32(float *data, float *dst, int ele_num);
void Relu6Fp32(float *data, float *dst, int ele_num);
#ifdef ENABLE_AVX
#ifdef WIN32
void ReluFp32C8(float *data, float *dst, int ele_num);
void Relu6Fp32C8(float *data, float *dst, int ele_num);
#endif
#endif
int offset(const int *shape, const int dim0, const int dim1, const int dim2, const int dim3);
int offsetComm(const int *shape, const int dim0, const int dim1, const int dim2);
int offset4d(const int *shape, const int *dims);

static inline bool isAddOverflow(int32_t x, int32_t y) {
  int32_t sum = x + y;
  return (x > 0 && y > 0 && sum < 0) || (x < 0 && y < 0 && sum > 0);
}

static inline bool isMulOverflow(int32_t x, int32_t y) {
  int32_t p = x * y;
  return (x != 0) && (p / x != y);
}

static inline int GetStride(int *strides, const int *shape, int length) {
  if (length <= 0) {
    return 1;
  }
  int stride = 1;
  for (int i = length - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return stride;
}

#ifdef ENABLE_ARM64
void BiasAdd(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu6(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu(const float *bias, float *data, size_t oc4, size_t plan_size);
void Relu6(float *data, size_t element4);
void Relu(float *data, size_t element4);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MINDSPORE_LITE_NNACL_COMMON_FUNC_H_ */
