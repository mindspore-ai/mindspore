/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <math.h>
#include <string.h>
#include "nnacl/fp32_grad/batch_norm.h"

void sumSpatialBatch(const float *in, size_t size, int ch, float *out) {
  memset(out, 0, ch * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    const float *ptr = in + (i * ch);
    for (size_t c = 0; c < ch; c++) {
      out[c] += ptr[c];
    }
  }
}

void backwardX(const float *in, const float *dout, const float *scale, const size_t size, int channels, float *mean,
               float *invar, float *dxhathat_sum, float *dxhat_sum, float *out) {
  const float N = (size);
  for (size_t i = 0; i < size; i++) {
    for (size_t f = 0; f < channels; f++) {
      size_t ix = i * channels + f;
      float x_hat = (in[ix] - mean[f]) * invar[f];
      float dx_hat = dout[ix] * scale[f];
      dxhat_sum[f] += dx_hat;
      dxhathat_sum[f] += dx_hat * x_hat;
    }
  }
  for (size_t i = 0; i < size; i++) {
    for (size_t f = 0; f < channels; f++) {
      size_t ix = i * channels + f;
      float x_hat = (in[ix] - mean[f]) * invar[f];
      float dx_hat = dout[ix] * scale[f];
      out[ix] = 1.0f / N * (invar[f]) * (N * dx_hat - dxhat_sum[f] - x_hat * dxhathat_sum[f]);
    }
  }
}

void backwardScale(const float *x, const float *mean, const float *invar, const float *delta, int batch, int n,
                   int size, float *scale_updates) {
  size_t i, b, f;
  memset(scale_updates, 0, n * sizeof(float));
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < size; ++i) {
      for (f = 0; f < n; ++f) {
        int index = (b * size + i) * n + f;
        float x_norm = (x[index] - mean[f]) * invar[f];
        scale_updates[f] += (delta[index] * x_norm);
      }
    }
  }
}

void var2Invar(float *save_var, size_t size, float eps) {
  for (size_t i = 0; i < size; i++) {
    save_var[i] = 1.0f / sqrt(save_var[i] + eps);
  }
}
