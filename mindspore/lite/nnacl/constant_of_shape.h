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
#ifndef MINDSPORE_LITE_NNACL_CONSTANT_OF_SHAPE_H_
#define MINDSPORE_LITE_NNACL_CONSTANT_OF_SHAPE_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

typedef struct ConstantOfShapeParameter {
  OpParameter op_parameter_;
  union value_ {
    float f32_value_;
    int32_t int32_value_;
  } value_;
  int data_type_;
  int element_size_;
} ConstantOfShapeParameter;

#ifdef __cplusplus
extern "C" {
#endif
int ConstantOfShapeFp32(float *output, int start, int end, float value);
int ConstantOfShapeInt32(int32_t *output, int start, int end, int32_t value);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_CONSTANT_OF_SHAPE_H_
