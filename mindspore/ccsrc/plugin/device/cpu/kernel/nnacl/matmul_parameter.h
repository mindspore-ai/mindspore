/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_MATMUL_H_
#define MINDSPORE_NNACL_MATMUL_H_

#include "nnacl/op_base.h"

typedef void (*MATMUL_OPT_R4_FUNC)(const int8_t *a, const int8_t *b, int32_t *dst, int row_4, int col_4, int deep_16,
                                   const int32_t *input_sum, const int32_t *bias);

typedef void (*MATMUL_OPT_R_FUNC)(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                  size_t stride, const int32_t *input_sum, const int32_t *bias,
                                  const int32_t *left_shift, const int32_t *right_shift, const int32_t *multiplier,
                                  int32_t output_zp, int32_t mini, int32_t maxi, size_t per_channel);

typedef void (*MATMUL_OPT_DP_FUNC)(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                   size_t stride, const int32_t *input_sum, const int32_t *bias,
                                   const int32_t *left_shift, const int32_t *right_shift, const int32_t *multiplier,
                                   int32_t output_zp, int32_t mini, int32_t maxi, size_t per_channel,
                                   const int32_t *filter_zp);

typedef enum OutType { OutType_C8 = 0, OutType_Nhwc = 1, OutType_TileC8 = 2, OutType_NC4HW4 = 3 } OutType;

typedef enum MatmulType {
  // reserve 0 for base op
  kNotImplemented = 0,
  kMatmulInt8Cpu,
  kMatmulDynamicInt8Cpu,
  kMatmulDynamicSdotInt8Cpu,
  kMatmulFp32BaseCpu,
  kMatmulFp32Arm64Cpu,
} MatmulType;

typedef struct MatMulParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  bool has_bias_;

  // other parameter
  int row_;
  int col_;
  int row_4_;
  int row_6_;
  int row_12_;
  int row_16_;
  int row_align_;
  int col_4_;
  int col_8_;
  int col_align_;
  int deep_;
  int deep_4_;
  int deep_16_;
  int deep_align_;
  int batch;
  bool a_transpose_; /* false :  row-major  */
  bool b_transpose_; /* true  :  col-major  */
  bool a_const_;
  bool b_const_;
  ActType act_type_;
  bool use_axis_;
  int axis_;
  MatmulType matmul_type_;
} MatMulParameter;

typedef struct MatmulQuantParameter {
  QuantArg input_;
  QuantArg weight_;
  QuantArg output_;
  int32_t out_act_min_;
  int32_t out_act_max_;
  float *filter_scale_;
  int32_t *filter_zp_;
  int32_t *left_shift_;
  int32_t *right_shift_;
  int32_t *quant_multiplier_;
} MatmulQuantParameter;

typedef struct MatmulDynamicQuantParameter {
  float input_scale_;
  int32_t input_zp_;
  float *filter_scale_;
  int32_t *filter_zp_;
} MatmulDynamicQuantParameter;

#endif  // MINDSPORE_NNACL_MATMUL_H_
