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
#ifndef MINDSPORE_LITE_NNACL_FP16_CONSTANT_OF_SHAPE_FP16_H_
#define MINDSPORE_LITE_NNACL_FP16_CONSTANT_OF_SHAPE_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/constant_of_shape_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
#ifdef ENABLE_NEON
inline int ConstantOfShapeFp16(float16_t *output, int start, int end, float16_t value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}
#endif
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_CONSTANT_OF_SHAPE_FP16_H_
