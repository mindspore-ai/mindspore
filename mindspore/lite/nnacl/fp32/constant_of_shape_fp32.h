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
#ifndef MINDSPORE_LITE_NNACL_FP32_CONSTANT_OF_SHAPE_FP32_H_
#define MINDSPORE_LITE_NNACL_FP32_CONSTANT_OF_SHAPE_FP32_H_
#include <memory.h>
#include <float.h>
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/constant_of_shape_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
inline int ConstantOfShapeInt32(int32_t *output, int start, int end, int32_t value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}

inline int ConstantOfShapeFp32(float *output, int start, int end, float value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_CONSTANT_OF_SHAPE_FP32_H_
