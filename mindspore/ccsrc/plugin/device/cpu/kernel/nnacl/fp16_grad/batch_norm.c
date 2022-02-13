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
#include <math.h>
#include <string.h>
#include "nnacl/fp16_grad/batch_norm.h"

void var2InvarFp16(float16_t *save_var, int size, float eps) {
  for (int i = 0; i < size; i++) {
    save_var[i] = (float16_t)(1.0f / sqrtf((float)save_var[i] + eps));
  }
}

void backwardAllFp16(const float16_t *restrict in, const float16_t *restrict yt, const float16_t *restrict mean,
                     const float16_t *restrict invar, const float16_t *restrict scale, int size, int ch,
                     float *restrict dxhat_sum, float *restrict dxhathat_sum, float16_t *restrict dbias,
                     float16_t *restrict dscale, float16_t *restrict dx) {
  NNACL_CHECK_ZERO_RETURN(size);
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      int ix = i * ch + c;
      dbias[c] += yt[ix];
      // dscale
      float16_t x_hat = (in[ix] - mean[c]) * invar[c];
      dscale[c] += (yt[ix] * x_hat);
      // dx_1
      float dx_hat = (float)(yt[ix] * scale[c]);
      dxhat_sum[c] += dx_hat;
      dxhathat_sum[c] += (float)(dx_hat * x_hat);
    }
  }
  float N = (float)size;
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      // dx_2
      int ix = i * ch + c;
      float16_t x_hat = (in[ix] - mean[c]) * invar[c];
      float16_t dx_hat = yt[ix] * scale[c];
      dx[ix] = (float16_t)((float)((invar[c]) * (N * dx_hat - dxhat_sum[c] - x_hat * dxhathat_sum[c])) / N);
    }
  }
}
void backwardP1Fp16(const float16_t *restrict in, const float16_t *restrict yt, const float16_t *restrict mean,
                    const float16_t *restrict invar, const float16_t *restrict scale, int size, int ch,
                    float *restrict dxhat_sum, float *restrict dxhathat_sum, float16_t *restrict dbias,
                    float16_t *restrict dscale) {
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      int ix = i * ch + c;
      dbias[c] += yt[ix];
      // dscale
      float x_hat = (float)((in[ix] - mean[c]) * invar[c]);
      dscale[c] += (yt[ix] * x_hat);
      // dx_1
      float dx_hat = (float)(yt[ix] * scale[c]);
      dxhat_sum[c] += dx_hat;
      dxhathat_sum[c] += dx_hat * x_hat;
    }
  }
}

void backwardP2Fp16(const float16_t *restrict in, const float16_t *restrict yt, const float16_t *restrict mean,
                    const float16_t *restrict invar, const float16_t *restrict scale, int size, int total_size, int ch,
                    const float *dxhat_sum, const float *dxhathat_sum, float16_t *restrict dx) {
  NNACL_CHECK_ZERO_RETURN(total_size);
  const float N = (float)total_size;
  for (int i = 0; i < size; i++) {
    for (int c = 0; c < ch; c++) {
      // dx_2
      int ix = i * ch + c;
      float x_hat = (float)((in[ix] - mean[c]) * invar[c]);
      float dx_hat = (float)(yt[ix] * scale[c]);
      dx[ix] = (float16_t)(((float)(invar[c]) * (N * dx_hat - dxhat_sum[c] - x_hat * dxhathat_sum[c])) / N);
    }
  }
}
