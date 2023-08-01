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
#ifndef NNACL_POOLING_PARAMETER_H_
#define NNACL_POOLING_PARAMETER_H_

#include "nnacl/op_base.h"

typedef enum PoolMode { PoolMode_No, PoolMode_MaxPool, PoolMode_AvgPool } PoolMode;

typedef enum RoundType { RoundType_No, RoundType_Ceil, RoundType_Floor } RoundType;

typedef struct PoolingParameter {
  OpParameter op_parameter_;
  PoolMode pool_mode_;
  RoundType round_type_;
  PadType pad_mode_;
  ActType act_type_;
  int avg_mode_;
  bool global_;
  int window_w_;
  int window_h_;
  int stride_w_;
  int stride_h_;
  int pad_u_;
  int pad_d_;
  int pad_l_;
  int pad_r_;
} PoolingParameter;

typedef struct Pooling3DParameter {
  PoolingParameter pooling_parameter_;
  int window_d_;
  int stride_d_;
  int input_d_;
  int output_d_;
  int pad_f_;  // front
  int pad_b_;  // back
  bool count_include_pad_;
  int divisor_override_;
} Pooling3DParameter;

#endif  // NNACL_POOLING_PARAMETER_H_
