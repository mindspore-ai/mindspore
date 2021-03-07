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
#ifndef MINDSPORE_LITE_NNACL_LAYER_NORM_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_LAYER_NORM_PARAMETER_H_

#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"

enum ElementwiseMode { ELEMENTWISE_NOT = 0, ELEMENTWISE_PER_CHANNEL = 1, ELEMENTWISE_PER_NUM = 2 };
typedef struct LayerNormParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  float epsilon_;
  enum ElementwiseMode elementwise_mode_;
  bool elementwise_affine_;
  int begin_norm_axis_;
  int begin_params_axis_;
  // shape correlative
  int norm_inner_size_;
  int norm_outer_size_;
  int params_inner_size_;
  int params_outer_size_;
  int normalized_dims_;
  int normalized_shape_[MAX_SHAPE_SIZE];
  // other parameter
  int thread_count_;
  int thread_outsize_;
} LayerNormParameter;

typedef struct LayerNormQuantArg {
  int32_t in_zp_;
  int32_t out_zp_;
  double in_scale_;
  double out_scale_;
} LayerNormQuantArg;

#endif  // MINDSPORE_LITE_NNACL_LAYER_NORM_PARAMETER_H_
