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
#include "nnacl/experimental/conv_fp32_nchwx_avx512.h"
#include "nnacl/experimental/conv.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/infer/conv2d_infer.h"
// #include "nnacl/intrinsics/ms_simd_avx512_instructions.h"

static const int UNIT_LEN = 512;  // register length

#define UNIT_NR 128  // (512 / sizeof(float))
static const int TILE_ROW;
int conv2d_prepare_fp32_nchwx_avx512(struct KernelBase *self) {
  KConv2d *conv = (KConv2d *)self;

  int rowIndex = 0;
  TensorC *weight = &(self->in[kWeightIndex]);

  int cout = weight->shape_[kNCHW_N];
  int cin = weight->shape_[kNCHW_C];
  int kh = weight->shape_[kNCHW_H];
  int kw = weight->shape_[kNCHW_W];

  size_t lineLen = cin * kw * kh;
  conv->packedWeight = malloc(lineLen * UP_DIV(cout, UNIT_NR) * sizeof(float));  // allocate packed weight buf

  float *rpos[16] = {0};
  float *data = (float *)weight->data_;
  float *buf = (float *)conv->packedWeight;
  int pos = 0;
  int bufIndex = 0;
  // transpose the weight matrix(width = cin*kw*kw, height = cout) from z order to z-N-Z order, z height of z-N-Z order
  // is UNIT and z width of z-N-Z order is UNIT, if not aligned, pad with 0
  while (rowIndex < cout) {
#ifdef VECTORIZE_OPTIMIZE
// use AVX2 instruction to optimize matrix transpose
#else
    for (int r = 0; r < 16 && (rowIndex + r) < cout; r++) {
      rpos[r] = data + pos;
      pos += lineLen;
    }
    for (int c = 0; c < lineLen; c++) {
      for (int r = 0; r < 16; r++) {
        if ((rowIndex + r) < cout) {
          buf[bufIndex] = *(rpos[r] + c);
        } else {
          buf[bufIndex] = 0;
        }
        bufIndex++;
      }
    }
#endif
    rowIndex += 16;
  }
  return 0;
}
int conv2d_release_fp32_nchwx_avx512(struct KernelBase *self) {
  KConv2d *conv = (KConv2d *)self;
  free(conv->im2colBuf);
  free(conv->packedWeight);
  return 0;
}
// position map from z order to n-Z order, srcw: src width, nh: n height
int PosMapz2nZ(int srcOffset, int srcw, int nh) {
  int groupSize = srcw * nh;
  int remain = srcOffset % groupSize;

  int dstX = remain % srcw;
  int dstY = remain / srcw;

  int dstOffset = groupSize - remain + dstX * nh + dstY;
  return dstOffset;
}

int conv2d_compute_fp32_nchwx_avx512(struct KernelBase *self) {
  KConv2d *conv = (KConv2d *)self;
  ConvParameter *param = (ConvParameter *)self->param;
  TensorC *in = &(self->in[kInputIndex]);
  TensorC *weight = &(self->in[kWeightIndex]);
  TensorC *out = &(self->out[kOutputIndex]);

  float *weightData = (float *)weight->data_;

  // im2col & tiling & pack
  float *buf = (float *)conv->im2colBuf;
  float *data = (float *)in->data_;
  int fmw = in->shape_[kNCHW_W] + param->pad_l_ + param->pad_r_;
  int fmh = in->shape_[kNCHW_H] + param->pad_u_ + param->pad_d_;
  int kh = weight->shape_[kNCHW_H];
  int kw = weight->shape_[kNCHW_W];
  int ci = UP_ROUND(in->shape_[kNCHW_C], 16);
  int co = UP_ROUND(out->shape_[kNCHW_C], 16);

  // tiling policy
  // m, n, k are the the left/right tile's shape
  int m = TILE_ROW;
  int n = UNIT_NR;
  int k = UNIT_NR;

  // im2col + pack to z-N-Z order
  int unitOffset = 0;
  int unitChNr = in->shape_[1];
  int interval = in->shape_[1] * UNIT_LEN;
#ifdef VECTORIZE_OPTIMIZE
// use AVX2 instruction to optimize matrix transpose
#else
  for (int wpos = 0; wpos < fmw - kw; wpos++) {
    for (int hpos = 0; hpos < fmh - kh; hpos++) {
      for (int x = 0; x < kw; x++) {
        for (int y = 0; y < kh; y++) {
          if ((wpos + x) < param->pad_l_ || (wpos + x) >= in->shape_[kNCHW_W] + param->pad_l_ ||
              (hpos + y) < param->pad_u_ || (hpos + y) >= in->shape_[kNCHW_H] + param->pad_d_) {
            memset(buf + PosMapz2nZ(unitOffset, unitChNr, m) * UNIT_LEN, 0, UNIT_LEN);
            unitOffset++;
          } else {
            int fmx = wpos + x - param->pad_l_;
            int fmy = hpos + y - param->pad_u_;
            int fmpos = (fmx * interval * in->shape_[kNCHW_W] + fmy * interval) * sizeof(float);
            // copy the whole channel for this feature map position
            for (int ch = 0; ch < unitChNr; ch++) {
              // transpose the feature map (width:cin*kw*kh, height:outh*outw, ) from z order to z-N-Z order, z width of
              // z-N-Z order is UNIT, z height of z-N-Z order is TILE_ROW. if not aligned, pad with 0.
              memcpy(buf + PosMapz2nZ(unitOffset, m, unitChNr) * UNIT_LEN,
                     data + fmpos + ch * interval * in->shape_[kNCHW_W] * in->shape_[kNCHW_H], UNIT_LEN);
              unitOffset++;
            }
          }
        }
      }
    }
  }
#endif
  // gemm: left matrix is feature map with z-N-Z order, right matrix is kernel with z-N-Z order, compute left
  // matrix multiple with the transpse of the right matrix
  int lmw = ci * kw * kh;                                 // left matrix width
  int lmh = out->shape_[kNCHW_H] * out->shape_[kNCHW_W];  // left matrix height
  // int rmw = lmw;
  int rmh = co;

#ifdef VECTORIZE_OPTIMIZE
// use AVX2 instruction to optimize gemm
#else

  float intputReg[C16NUM][UNIT_NR];
  float weightReg[C16NUM][UNIT_NR];
  float outputReg[C16NUM][UNIT_NR];
  memset(outputReg, 0, sizeof(outputReg));
  int lpos = 0;
  int rpos = 0;
  int outOffset = 0;
  for (int x = 0; x < rmh; x++) {    // output channel
    for (int y = 0; y < lmh; y++) {  // output h * w
      // tile computing
      for (int tilePos = 0; tilePos < lmw; tilePos++) {
        // load left tile
        int regNr = 0;
        for (int c = 0; c < m; c++) {
          memcpy(&intputReg[regNr++], buf + lpos, UNIT_LEN);
          lpos += UNIT_LEN;
        }
        // load right tile
        regNr = 0;
        for (int c = 0; c < k; c++) {
          memcpy(&weightReg[regNr++], weightData + rpos, UNIT_LEN);
          rpos += UNIT_LEN;
        }

        // matrix multiplication: [m,n] * [n,k]
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < k; j++) {
            for (int p = 0; p < n; p++) {
              outputReg[i][j] += intputReg[i][p] * weightReg[j][p];
            }
          }
        }
        tilePos += n;
      }
      // flush outputReg to output tensor memory
      memcpy((float *)out->data_ + outOffset, outputReg, m * k * sizeof(float));
      outOffset += m * k * sizeof(float);
      y += m;
    }
    x += k;
  }
#endif
  return 0;
}

int conv2d_infershape_fp32_nchwx_avx512(struct KernelBase *self) {
  return Conv2dInferShape((const struct TensorC *const *)(&(self->in)), self->insize, &(self->out), self->outsize,
                          self->param);
}

int conv2d_resize_fp32_nchwx_avx512(struct KernelBase *self) {
  KConv2d *conv = (KConv2d *)self;

  TensorC *in = &(self->in[kInputIndex]);
  TensorC *weight = &(self->in[kWeightIndex]);
  TensorC *out = &(self->out[kOutputIndex]);
  int kh = weight->shape_[kNCHW_H];
  int kw = weight->shape_[kNCHW_W];

  self->inferShape(self);
  out->format_ = Format_NC16HW16;
  out->shape_[1] = UP_DIV(out->shape_[1], C16NUM);
  out->shape_[4] = 16;
  out->shape_size_ = 5;

  if (conv->im2colBuf) {
    free(conv->im2colBuf);
  }
  int ci = in->shape_[1] * in->shape_[4];
  int lmw = ci * kw * kh;                                 // left matrix width
  int lmh = out->shape_[kNCHW_H] * out->shape_[kNCHW_W];  // left matrix height

  conv->im2colBuf = malloc(lmw * lmh * sizeof(float));  // allocate im2col buf
  return 0;
}
