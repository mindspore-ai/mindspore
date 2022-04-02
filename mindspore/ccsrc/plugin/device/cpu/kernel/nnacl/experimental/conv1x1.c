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

#include "nnacl/experimental/conv1x1.h"
#include <stdint.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/experimental/base_matmul.h"

typedef struct Conv1x1Stru {
  KernelBase base;
  uint8_t *bias_;
  uint8_t *weight_;
} Conv1x1Stru;

int conv1x1_resize(struct KernelBase *self, TensorC *in[], size_t insize, TensorC *out[], size_t outsize) { return 0; }

int conv1x1_prepare(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  ConvParameter *param = (ConvParameter *)conv->base.param;

  conv->base.funcs = GetCoreFuncs(conv->base.in[0]->data_type_ == kNumberTypeFloat16);

  int row_tile, deep_tile, col_tile;
  conv->base.funcs->InitMatmulTileCount(&row_tile, &deep_tile, &col_tile);

  conv->weight_ = (uint8_t *)(conv->base.env->alloc(
    conv->base.env->allocator,
    UP_ROUND(param->output_channel_, col_tile) * UP_ROUND(param->input_channel_, deep_tile) * row_tile));
  conv->base.funcs->PackRight(conv->base.in[1]->data_, conv->weight_, 1, param->input_channel_, param->output_channel_);

  if (conv->base.insize < kInputSize2) {
    conv->bias_ = NULL;
    return 0;
  }

  size_t bias_size = UP_ROUND(param->output_channel_, conv->base.funcs->pack) * conv->base.funcs->byte;
  conv->bias_ = (uint8_t *)(conv->base.env->alloc(conv->base.env->allocator, bias_size));

  memset(conv->bias_, 0, bias_size);
  memcpy(conv->bias_, conv->base.in[kBiasIndex]->data_, param->output_channel_);

  return 0;
}

int conv1x1_release(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  conv->base.env->free(conv->base.env->allocator, conv->bias_);
  conv->base.env->free(conv->base.env->allocator, conv->weight_);
  return 0;
}

int conv1x1_compute(struct KernelBase *self) {
  Conv1x1Stru *conv = (Conv1x1Stru *)self;
  ConvParameter *param = (ConvParameter *)conv->base.param;

  BaseMatmul(conv->base.in[0]->data_, conv->weight_, conv->bias_, conv->base.out[0]->data_,
             param->input_h_ * param->input_w_, param->input_channel_, param->output_channel_, param->act_type_,
             param->op_parameter_.thread_num_, &conv->base);
  return 0;
}

KernelBase *CreateConv1x1(OpParameter *param, TensorC **in, size_t insize, TensorC **out, size_t outsize) {
  if (in[0]->format_ != Format_NC4HW4) {
    return NULL;
  }
  Conv1x1Stru *conv1x1 = (Conv1x1Stru *)malloc(sizeof(Conv1x1Stru));
  conv1x1->base.param = param;
  conv1x1->base.in = in;
  conv1x1->base.insize = insize;
  conv1x1->base.out = out;
  conv1x1->base.outsize = outsize;
  conv1x1->base.env = GetExecEnv();
  conv1x1->base.prepare = conv1x1_prepare;
  conv1x1->base.resize = conv1x1_resize;
  conv1x1->base.release = conv1x1_release;
  conv1x1->base.compute = conv1x1_compute;

  return (KernelBase *)conv1x1;
}
