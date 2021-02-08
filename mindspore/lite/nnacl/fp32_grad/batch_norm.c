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

void var2Invar(float *save_var, int size, float eps) {
  for (int i = 0; i < size; i++) {
    save_var[i] = 1.0f / sqrt(save_var[i] + eps);
  }
}

void backwardAll(const float *restrict in, const float *restrict yt, const float *restrict mean,
                 const float *restrict invar, const float *restrict scale, int size, int ch, float *restrict dxhat_sum,
                 float *restrict dxhathat_sum, float *restrict dbias, float *restrict dscale, float *restrict dx) {
  float N = (float)size;
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      int ix = i * ch + c;
      dbias[c] += yt[ix];
      // dscale
      float x_hat = (in[ix] - mean[c]) * invar[c];
      dscale[c] += (yt[ix] * x_hat);
      // dx_1
      float dx_hat = yt[ix] * scale[c];
      dxhat_sum[c] += dx_hat;
      dxhathat_sum[c] += dx_hat * x_hat;
    }
  }
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      // dx_2
      int ix = i * ch + c;
      float x_hat = (in[ix] - mean[c]) * invar[c];
      float dx_hat = yt[ix] * scale[c];
      dx[ix] = 1.0f / N * (invar[c]) * (N * dx_hat - dxhat_sum[c] - x_hat * dxhathat_sum[c]);
    }
  }
}
void backwardP1(const float *restrict in, const float *restrict yt, const float *restrict mean,
                const float *restrict invar, const float *restrict scale, int size, int ch, float *restrict dxhat_sum,
                float *restrict dxhathat_sum, float *restrict dbias, float *restrict dscale) {
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      int ix = i * ch + c;
      dbias[c] += yt[ix];
      // dscale
      float x_hat = (in[ix] - mean[c]) * invar[c];
      dscale[c] += (yt[ix] * x_hat);
      // dx_1
      float dx_hat = yt[ix] * scale[c];
      dxhat_sum[c] += dx_hat;
      dxhathat_sum[c] += dx_hat * x_hat;
    }
  }
}

void backwardP2(const float *restrict in, const float *restrict yt, const float *restrict mean,
                const float *restrict invar, const float *restrict scale, int size, int total_size, int ch,
                const float *dxhat_sum, const float *dxhathat_sum, float *restrict dx) {
  const float N = (float)total_size;
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      // dx_2
      int ix = i * ch + c;
      float x_hat = (in[ix] - mean[c]) * invar[c];
      float dx_hat = yt[ix] * scale[c];
      dx[ix] = 1.0f / N * (invar[c]) * (N * dx_hat - dxhat_sum[c] - x_hat * dxhathat_sum[c]);
    }
  }
}
