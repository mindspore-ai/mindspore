/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef NNACL_KERNEL_LAYER_NORM_H_
#define NNACL_KERNEL_LAYER_NORM_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct LayerNormComputeParam {
  float epsilon_;
  bool elementwise_affine_;
  int begin_norm_axis_;
  int begin_params_axis_;
  int norm_inner_size_;
  int norm_outer_size_;
  int params_inner_size_;
  int params_outer_size_;
} LayerNormComputeParam;

typedef struct LayerNormStruct {
  KernelBase base_;
  LayerNormComputeParam compute_;
  int data_type_;
  void *src_data_;
  void *dst_data_;
  void *gamma_data_;
  void *beta_data_;
  void *mean_data_;
  void *var_data_;
} LayerNormStruct;

KernelBase *CreateLayerNorm(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_LAYER_NORM_H_
