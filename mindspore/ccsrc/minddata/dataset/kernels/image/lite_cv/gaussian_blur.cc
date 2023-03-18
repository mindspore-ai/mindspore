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

#include <cmath>

#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

#ifdef ENABLE_ANDROID
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#define USE_NEON
#include <arm_neon.h>
#endif
#endif

namespace mindspore {
namespace dataset {
static void GetGaussianKernel(float *kernel, int size, double sigma) {
  int n = (size - 1) / 2;
  std::vector<float> buffer(n);
  float sum = 0;
  for (int i = 0; i < n; i++) {
    int x = i - n;
    float g = exp(-0.5 * x * x / (sigma * sigma));
    buffer[i] = g;
    sum += g;
  }
  sum = sum * 2 + 1;
  if (size % 2 == 0) {
    sum += 1;
  }

  const float scale = 1. / sum;
  for (int i = 0; i < n; i++) {
    float g = buffer[i] * scale;
    kernel[i] = g;
    kernel[size - 1 - i] = g;
  }
  kernel[n] = scale;
  if (size % 2 == 0) {
    kernel[n + 1] = scale;
  }
}

bool GaussianBlur(const LiteMat &src, LiteMat &dst, const std::vector<int> &ksize, double sigmaX,  // NOLINT
                  double sigmaY, PaddBorderType pad_type) {
  if (src.IsEmpty() || src.data_type_ != LDataType::UINT8) {
    return false;
  }
  if (ksize.size() != 2 || ksize[0] <= 0 || ksize[1] <= 0 || ksize[0] % 2 != 1 || ksize[1] % 2 != 1) {
    return false;
  }
  if (sigmaX <= 0) {
    return false;
  }
  if (sigmaY <= 0) {
    sigmaY = sigmaX;
  }
  if (ksize[0] == 1 && ksize[1] == 1) {
    dst = src;
    return true;
  }

  LiteMat kx, ky;
  kx.Init(ksize[0], 1, 1, LDataType::FLOAT32);
  ky.Init(1, ksize[1], 1, LDataType::FLOAT32);
  RETURN_FALSE_IF_LITEMAT_EMPTY(kx);
  RETURN_FALSE_IF_LITEMAT_EMPTY(ky);

  GetGaussianKernel(kx, ksize[0], sigmaX);
  GetGaussianKernel(ky, ksize[1], sigmaY);

  return ConvRowCol(src, kx, ky, dst, src.data_type_, pad_type);
}
}  // namespace dataset
}  // namespace mindspore
