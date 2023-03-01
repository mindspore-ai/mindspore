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

int conv1x1_exp_resize(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  ConvParameter *param = (ConvParameter *)conv->base.param;
  conv->exp_.row = param->input_h_ * param->input_w_;
  conv->exp_.deep = param->input_channel_;
  conv->exp_.col = param->output_channel_;
  conv->exp_.thread_num = param->op_parameter_.thread_num_;
  if (conv->bias_ != NULL || param->act_type_ != ActType_No) {
    conv->exp_.base->funcs->PostParam(param->act_type_, &conv->exp_.min, &conv->exp_.max);
  }
  return 0;
}

int conv1x1_exp_prepare(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  ConvParameter *param = (ConvParameter *)conv->base.param;
  conv->exp_.base = &conv->base;

  int row_tile, deep_tile, col_tile;
  conv->base.funcs->ExpMatmulTile(&row_tile, &deep_tile, &col_tile);

  conv->weight_ = (uint8_t *)(conv->base.env->alloc(
    conv->base.env->allocator,
    UP_ROUND(param->output_channel_, col_tile) * UP_ROUND(param->input_channel_, deep_tile) * row_tile));
  conv->base.funcs->PackNcX(conv->base.in[1].data_, conv->weight_, 1, param->input_channel_, param->output_channel_);

  if (conv->base.insize < kInputSize2) {
    conv->bias_ = NULL;
    return 0;
  }

  size_t bias_size = UP_ROUND(param->output_channel_, conv->base.funcs->pack) * conv->base.funcs->byte;
  conv->bias_ = (uint8_t *)(conv->base.env->alloc(conv->base.env->allocator, bias_size));

  memset(conv->bias_, 0, bias_size);
  memcpy(conv->bias_, conv->base.in[kBiasIndex].data_, param->output_channel_);

  return 0;
}

int conv1x1_exp_release(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  conv->base.env->free(conv->base.env->allocator, conv->bias_);
  conv->base.env->free(conv->base.env->allocator, conv->weight_);
  return 0;
}

int conv1x1_exp_compute(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  ExperimentalMatmul(conv->base.in[0].data_, conv->weight_, conv->bias_, conv->base.out[0].data_, &conv->exp_);
  return 0;
}

KernelBase *CreateConv1x1(OpParameter *param, int data_type, FormatC format) {
  if (format == Format_NHWC) {
    return NULL;
  }

  ConvParameter *conv_param = (ConvParameter *)param;
  if (conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1) {
    return NULL;
  }

  Conv1x1Stru *conv1x1 = (Conv1x1Stru *)malloc(sizeof(Conv1x1Stru));
  conv1x1->base.funcs = GetCoreFuncs(data_type == kNumberTypeFloat16);
  conv1x1->base.prepare = conv1x1_exp_prepare;
  conv1x1->base.resize = conv1x1_exp_resize;
  conv1x1->base.release = conv1x1_exp_release;
  conv1x1->base.compute = conv1x1_exp_compute;

  return (KernelBase *)conv1x1;
}
