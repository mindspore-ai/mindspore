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

void sumSpatialBatch(const float *in, int size, int ch, float *out) {
  memset(out, 0, ch * sizeof(float));
  for (int i = 0; i < size; i++) {
    const float *ptr = in + i * ch;
    for (int c = 0; c < ch; c++) {
      out[c] += ptr[c];
    }
  }
}

static void meanVar(const float *in, int size, int ch, float eps, float *mean, float *invar) {
  float N = (float)(size);
  sumSpatialBatch(in, N, ch, mean);
  for (int f = 0; f < ch; ++f) {
    mean[f] /= N;
  }
  for (int f = 0; f < ch; f++) {
    float tvar = 0;
    for (int i = 0; i < N; i++) {
      float x = in[i * ch + f];
      tvar += (x - mean[f]) * (x - mean[f]);
    }
    invar[f] = 1.0f / (sqrt(tvar / N + eps));
  }
}

void backwardX(const float *in, const float *dout, const float *scale, const int size, int channels, float eps,
               float *mean, float *invar, float *dxhathat_sum, float *dxhat_sum, float *out) {
  meanVar(in, size, channels, eps, mean, invar);
  for (int i = 0; i < size; i++) {
    for (int f = 0; f < channels; f++) {
      int ix = i*channels + f;
      float x_hat = (in[ix] - mean[f]) * invar[f];
      float dxhat = dout[ix] * scale[f];
      dxhat_sum[f] += dxhat;
      dxhathat_sum[f] += dxhat * x_hat;
    }
  }
  for (int i = 0; i < size; i++) {
    for (int f = 0; f < channels; f++) {
      int ix = i*channels + f;
      float x_hat = (in[ix] - mean[f]) * invar[f];
      float dxhat = dout[ix] * scale[f];
      out[ix] = 1.f / size * invar[f] * (size * dxhat - dxhat_sum[f] - x_hat * dxhathat_sum[f]);
    }
  }
}

void backwardScale(const float *x, const float *mean, const float *invar, const float *delta, int batch,
                   int n, int size, float *scale_updates) {
  int i, b, f;
  memset(scale_updates, 0, n * sizeof(float));
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < size; ++i) {
      for (f = 0; f < n; ++f) {
        int index = (b * size + i) * n + f;
        float x_norm = (x[index] - mean[f]) * invar[f];
        scale_updates[f] += delta[index] * x_norm;
      }
    }
  }
}

