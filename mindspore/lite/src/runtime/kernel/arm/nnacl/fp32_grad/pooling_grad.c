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
#include <cstdint>
#include "nnacl/fp32_grad/pooling_grad.h"

void AvgPoolingGrad(const float *input_ptr, float *output_ptr, PoolingParameter *pooling_param) {
  int stride_w = pooling_param->stride_w_;
  int stride_h = pooling_param->stride_h_;
  int pad_w = pooling_param->pad_l_;
  int pad_h = pooling_param->pad_u_;
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int output_batch = pooling_param->output_batch_;

  const float *inPtr = NULL;
  for (int i = 0; i < output_h * output_w * channel * output_batch; i++) output_ptr[i] = 0.0;

  // int pad_top = padding[2];

  float kk = (float)(win_h * win_w);

  for (uint16_t ib = 0; ib < output_batch; ib++) {
    // int in_batch_offset = batch * in_h * in_w * channel;
    // int out_batch_offset = batch * output_h * output_w * channel;
    // out = grads->getData(ib*grads->imgSize());
    // inPtr = in->getData(ib*in->imgSize());
    float *out;
    out = &output_ptr[(ib * output_h * output_w)];
    inPtr = (float *)(&input_ptr[(ib * in_h * in_w)]);
    if (1) {  // in->layout() == Tensor::nhwc)
      // iterate over yt
      for (uint16_t yh = 0; yh < in_h; yh++) {
        for (uint16_t yw = 0; yw < in_w; yw++) {
          for (uint16_t ic = 0; ic < channel; ic++) {
            int idx = (yw + yh * in_w) * channel + ic;  // (ic*in_h*in_w) + (in_w*yh) + yw;
            float delta = inPtr[idx] / kk;
            for (int32_t kh = 0; kh < win_h; kh++) {
              int xh = yh * stride_h + kh - pad_h;
              if ((xh < 0) || (xh >= output_h)) {
                continue;
              }
              for (int32_t kw = 0; kw < win_w; kw++) {
                int xw = yw * stride_w + kw - pad_w;
                if ((xw < 0) || (xw >= output_w)) {
                  continue;
                }
                // out[(ic*output_h*output_w) + (xh*output_w) + xw] += delta;
                out[(xw + output_w * xh) * channel + ic] += delta;
              }
            }
          }
        }
      }
    } else {  // nchw
      for (uint16_t ic = 0; ic < channel; ic++) {
        // iterate over yt
        for (uint16_t yh = 0; yh < in_h; yh++) {
          for (uint16_t yw = 0; yw < in_w; yw++) {
            int idx = (ic * in_h * in_w) + (in_w * yh) + yw;
            float delta = inPtr[idx] / kk;
            for (int32_t kh = 0; kh < win_h; kh++) {
              int xh = yh * stride_h + kh - pad_h;
              if ((xh < 0) || (xh >= output_h)) {
                continue;
              }
              for (int32_t kw = 0; kw < win_w; kw++) {
                int xw = yw * stride_w + kw - pad_w;
                if ((xw < 0) || (xw >= output_w)) {
                  continue;
                }
                out[(ic * output_h * output_w) + (xh * output_w) + xw] += delta;
              }
            }
          }
        }
      }
    }
  }
}

void MaxPoolingGrad(const float *dy, const int *indices, float *output_ptr, PoolingParameter *pooling_param) {
  // int stride_w = pooling_param->stride_w_;
  // int stride_h = pooling_param->stride_h_;
  // int pad_w = pooling_param->pad_l_;
  // int pad_h = pooling_param->pad_u_;
  // int win_w = pooling_param->window_w_;
  // int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int output_batch = pooling_param->output_batch_;

  int out_img_size =
    output_h * output_w;  // Emir -- in original code this varible is calculated according to input size ??
  int ind_img_size = in_h * in_w;
  // const int w_pad = (output_w + pad_w + pad_w);

  for (int i = 0; i < output_h * output_w * channel * output_batch; i++) output_ptr[i] = 0.0;

  const float *yt = (const float *)(dy);
  const int *pos = (const int *)(indices);
  float *out = NULL;

  if (1) {  // grads->layout() == Tensor::nhwc)
    for (int ib = 0; ib < output_batch; ib++) {
      out = &(output_ptr[ib * output_w * output_w * channel]);
      for (int ix = 0; ix < ind_img_size; ix++) {
        for (int cix = 0; cix < channel; cix++) {
          int idx = (*pos) * channel + cix;
          out[idx] += *yt;
          pos++;
          yt++;
        }
      }
    }
  } else {
    for (int ib = 0; ib < output_batch; ib++) {
      out = &output_ptr[(ib * out_img_size)];
      for (int cix = 0; cix < channel; cix++) {
        for (int ix = 0; ix < ind_img_size; ix++) {
          int idx = cix * output_h * output_w + *pos;  // cord_y*output_w + cord_x;
          out[idx] += *yt;
          pos++;
          yt++;
        }
      }
    }
  }
}
