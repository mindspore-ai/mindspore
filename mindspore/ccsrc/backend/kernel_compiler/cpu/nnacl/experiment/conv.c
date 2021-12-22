/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/conv_parameter.h>
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/kernel.h"

typedef struct ConvStru {
  KernelStru base;
  int inIm2colW;
  int inIm2colH;
} ConvStru;

int conv_init_fp32_nc4hw4_armv8(struct KernelStru *self, KernelContext *ctx) {
  ConvStru *conv = (ConvStru *)self;
  self->ctx = ctx;
  self->infershape(self->param, self->in, self->insize, self->out, self->outsize);

  int outw = self->out[kOutputIndex]->shape_[kNCHW_W];
  int outh = self->in[kWeightIndex]->shape_[kNCHW_H];
  int inch = self->in[kInputIndex]->shape_[kNCHW_C];
  int kw = self->in[kWeightIndex]->shape_[kNCHW_W];
  int kh = self->in[kWeightIndex]->shape_[kNCHW_H];

  // im2col buffer
  conv->inIm2colW = inch * kw * kh;
  conv->inIm2colH = outw * outh;
  self->buf[0] = ctx->alloc(conv->inIm2colW * conv->inIm2colH);
  self->buf[1] = ctx->alloc(conv->inIm2colW);

  return 0;
}

int conv_release_fp32_nc4hw4_armv8(KernelStru *self) {
  size_t sz = sizeof(self->buf) / sizeof(self->buf[0]);
  for (size_t i = 0; i < sz; i++) {
    free(self->buf[sz]);
  }
  return 0;
}

int conv_compute_fp32_nc4hw4_armv8(KernelStru *self) {
  int outw = self->out[kOutputIndex]->shape_[kNCHW_W];
  int outh = self->in[kWeightIndex]->shape_[kNCHW_H];
  int outch = self->out[kOutputIndex]->shape_[kNCHW_C];
  int inw = self->in[kInputIndex]->shape_[kNCHW_W];
  int inh = self->in[kInputIndex]->shape_[kNCHW_H];
  int inch = self->in[kInputIndex]->shape_[kNCHW_C];
  int kw = self->in[kWeightIndex]->shape_[kNCHW_W];
  int kh = self->in[kWeightIndex]->shape_[kNCHW_H];

  int outPos = 0;
  float *outPtr = (float *)self->out[kOutputIndex]->data_;

  ConvParameter *param = (ConvParameter *)self->param;
  for (size_t n = 0; n < self->out[kOutputIndex]->shape_[kNCHW_N]; n++) {
    // im2col input
    float *inIm2colBuf = (float *)self->buf[0];
    int index = 0;

    // along the input height direction
    for (int y = 0; y < outh; y++) {
      // along the input width direction
      for (int x = 0; x < outw; x++) {
        // per input channel
        for (int ch = 0; ch < inch; ch++) {
          float *fp = (float *)(self->in[kInputIndex] + inch * inw * inh);

          // per sliding window
          for (int rowStart = 0; rowStart < kh; rowStart++) {
            for (int colStart = 0; colStart < kw; colStart++) {
              int posx = x + colStart;
              int posy = y + rowStart;

              // the padding area
              if (posx < inw || posx >= inw + param->pad_l_ || posy < inh || posy >= inh + param->pad_u_) {
                inIm2colBuf[index++] = 0;
                continue;
              }

              inIm2colBuf[index++] = *(fp + (posy - param->pad_u_) * inw + (posx - param->pad_l_));
            }
          }
        }
      }
    }

    for (size_t co = 0; co < outch; co++) {  // along out channel direction
      index = 0;
      float *wtIm2colBuf = self->buf[1];
      float *fp = (float *)(self->in[kWeightIndex] + co * inch * kh * kw);

      // im2col weight
      for (int ch = 0; ch < inch; ch++) {
        for (int y = 0; y < kh; y++) {
          for (int x = 0; x < kw; x++) {
            wtIm2colBuf[index++] = *(fp + ch * kh * kw + y * kh + x);
          }
        }
      }

      for (int y = 0; y < outh * outw; y++) {  // along output height*width direction
        float *rowBuf = inIm2colBuf + y * kw * kh;
        float *colBuf = wtIm2colBuf;
        float *outfp = outPtr + outPos;
        *outfp = 0;
        for (int l = 0; l < kh * kw; l++) {
          *outfp += rowBuf[l] * colBuf[l];
        }
        outPos++;
      }
    }
  }
  return 0;
}

static KernelStru *CreateConv(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize) {
  ConvStru *conv = (ConvStru *)malloc(sizeof(ConvStru));
  conv->base.init = conv_init_fp32_nc4hw4_armv8;
  conv->base.release = conv_release_fp32_nc4hw4_armv8;
  conv->base.compute = conv_compute_fp32_nc4hw4_armv8;
  conv->base.param = param;
  conv->base.in = in;
  conv->base.insize = insize;
  conv->base.out = out;
  conv->base.outsize = outsize;
  return (KernelStru *)conv;
}

REG_KERNEL_CREATOR(Conv2D, PrimType_Conv2DFusion, kDataTypeFloat, NC4HW4, CreateConv);
