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
#ifndef MINDSPORE_LITE_NNACL_ARITHMETIC_COMMON_H_
#define MINDSPORE_LITE_NNACL_ARITHMETIC_COMMON_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/arithmetic_common.h"

typedef struct ArithmeticParameter {
  OpParameter op_parameter_;
  bool broadcasting_;
  size_t ndim_;
  int activation_type_;
  int in_shape0_[10];
  int in_elements_num0_;
  int in_shape1_[10];
  int in_elements_num1_;

  int out_shape_[10];
  int out_elements_num_;

  int in_strides0_[10];
  int in_strides1_[10];
  int out_strides_[10];

  int multiples0_[10];
  int multiples1_[10];
} ArithmeticParameter;

#ifdef __cplusplus
extern "C" {
#endif
void TileOneDimension(const float *inData, float *outData, int dim, size_t ndim, const int *inShape,
                      const int *inStrides, const int *outStrides, const int *multiple);
void ComputeStrides(const int *shape, int *strides, const int ndim);

void CalcMultiplesAndStrides(ArithmeticParameter *param);

void TileOneDimensionUint8(const uint8_t *inData, uint8_t *outData, int dim, size_t ndim, const int *inShape,
                           const int *inStrides, const int *outStrides, const int *multiple);
void TileDimensions(const float *data0, const float *data1, float *tile_data0, float *tile_data1,
                    ArithmeticParameter *param);
void TileDimensionsUint8(const uint8_t *data0, const uint8_t *data1, uint8_t *tile_data0, uint8_t *tile_data1,
                         ArithmeticParameter *param);
void TileDimensionsInt8(const int8_t *data0, const int8_t *data1, int8_t *tile_data0, int8_t *tile_data1,
                        ArithmeticParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_ARITHMETIC_COMMON_H_
