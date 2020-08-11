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
#include <string.h>
#include <math.h>
#include "nnacl/fp32_grad/batch_norm.h"

static void sumSpatialBatch(const float *in, int size, int ch, float *out) {
  memset(out, 0, ch * sizeof(float));
  for (int i = 0; i < size; i++) {
    const float *ptr = in + i * ch;
    for (int c = 0; c < ch; c++) {
      out[c] += ptr[c];
    }
  }
}

void scaleBias(const float *scales, int batch, int n, int size, float *output) {
  for (int i = 0; i < batch * size; i++)
    for (int c = 0; c < n; c++) output[i * n + c] *= scales[c];
}

void normalize(const float *x, const float *mean, const float *variance, float eps, int batch, int filters, int spatial,
               float *out) {
  int b, f, i;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < spatial; ++i) {
      for (f = 0; f < filters; ++f) {
        int index = b * filters * spatial + i * filters + f;
        out[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + eps);
      }
    }
  }
}

void backwardScale(const float *x_norm, const float *delta, int batch, int n, int size, float *scale_updates) {
  int i, b, f;
  memset(scale_updates, 0, n * sizeof(float));
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < size; ++i) {
      for (f = 0; f < n; ++f) {
        int index = (b * size + i) * n + f;
        scale_updates[f] += delta[index] * x_norm[index];
      }
    }
  }
}

void meanVar(const float *in, int batch, int spatial, int ch, float *mean, float *var) {
  float N = batch * spatial;
  sumSpatialBatch(in, N, ch, mean);
  for (int f = 0; f < ch; ++f) mean[f] /= N;
  memset(var, 0, ch * sizeof(float));
  for (int i = 0; i < N; i++) {
    for (int f = 0; f < ch; f++) {
      float x = in[i * ch + f];
      var[f] += (x - mean[f]) * (x - mean[f]);
    }
  }
  for (int f = 0; f < ch; f++) var[f] /= N;
}

void meanDelta(float *yt, int size, int ch, float eps, float *variance, float *mean_delta) {
  sumSpatialBatch(yt, size, ch, mean_delta);
  for (int i = 0; i < ch; i++) mean_delta[i] *= -1.f / sqrt((variance[i] + eps));
}

void meanAdd(const float *x, const float *mean, const float *variance_delta, int batch, int filters, int spatial,
             float *mean_add, float *mean_delta) {
  int i, k;
  memset(mean_add, 0, filters * sizeof(float));
  for (k = 0; k < spatial * batch; ++k) {
    for (i = 0; i < filters; ++i) {
      int index = k * filters + i;
      mean_add[i] += x[index] - mean[i];
    }
  }
  for (i = 0; i < filters; ++i) {
    mean_add[i] *= variance_delta[i] * (-2.f / (spatial * batch));
    mean_delta[i] += mean_add[i];
  }
}

void varianceDelta(const float *x, const float *delta, const float *mean, const float *variance, int batch, int filters,
                   int spatial, float eps, float *variance_delta) {
  int i, k;
  memset(variance_delta, 0, filters * sizeof(float));
  for (k = 0; k < batch * spatial; k++) {
    for (i = 0; i < filters; i++) {
      int index = k * filters + i;
      variance_delta[i] += delta[index] * (x[index] - mean[i]);
    }
  }
  for (i = 0; i < filters; i++) variance_delta[i] *= -.5 * pow(variance[i] + eps, (-3.f / 2.f));
}

void NormalizeDelta(const float *x, const float *mean, const float *variance, const float *mean_delta,
                    const float *variance_delta, int batch, int filters, int spatial, float eps, float *delta) {
  int f, k;
  for (k = 0; k < batch * spatial; k++) {
    for (f = 0; f < filters; f++) {
      int index = k * filters + f;
      delta[index] = delta[index] * 1. / (sqrt(variance[f] + eps)) +
                     variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) +
                     mean_delta[f] / (spatial * batch);
    }
  }
}
