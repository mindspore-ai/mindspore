/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/convolution_1x1.h"
#include <stdint.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"

int conv1x1_exp_release(struct KernelBase *self) { return 0; }
int conv1x1_exp_prepare(struct KernelBase *self) { return 0; }
int conv1x1_exp_resize(struct KernelBase *self) { return 0; }
int conv1x1_exp_compute(struct KernelBase *self) { return 0; }

KernelBase *CreateConv1x1(OpParameter *param, int data_type) {
  ConvParameter *conv_param = (ConvParameter *)param;
  if (conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1) {
    return NULL;
  }

  Conv1x1Stru *conv1x1 = (Conv1x1Stru *)malloc(sizeof(Conv1x1Stru));
  conv1x1->base.prepare = conv1x1_exp_prepare;
  conv1x1->base.resize = conv1x1_exp_resize;
  conv1x1->base.release = conv1x1_exp_release;
  conv1x1->base.compute = conv1x1_exp_compute;

  return (KernelBase *)conv1x1;
}
