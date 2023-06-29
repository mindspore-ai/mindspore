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
#ifndef NNACL_LAYER_NORM_PARAMETER_H_
#define NNACL_LAYER_NORM_PARAMETER_H_

#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"

typedef struct LayerNormParameter {
  OpParameter op_parameter_;
  float epsilon_;
  bool elementwise_affine_;
  int begin_norm_axis_;
  int begin_params_axis_;
} LayerNormParameter;

typedef struct LayerNormQuantArg {
  int32_t in_zp_;
  int32_t out_zp_;
  double in_scale_;
  double out_scale_;
} LayerNormQuantArg;

#endif  // NNACL_LAYER_NORM_PARAMETER_H_
