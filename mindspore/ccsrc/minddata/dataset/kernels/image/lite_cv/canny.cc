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

#define ANGLE_22_5 0.39269908169872414
#define ANGLE_67_5 1.1780972450961724

namespace mindspore {
namespace dataset {
static void GetSobelKernel(float *kernel, int flag, int ksize, double scale) {
  std::vector<float> buffer(ksize + 1);

  if (ksize == 1) {
    buffer[0] = 1;
  } else if (ksize == 3) {
    if (flag == 0) {
      buffer[0] = 1, buffer[1] = 2, buffer[2] = 1;
    } else if (flag == 1) {
      buffer[0] = -1, buffer[1] = 0, buffer[2] = 1;
    } else {
      buffer[0] = 1, buffer[1] = -2, buffer[2] = 1;
    }
  } else {
    int old, now;
    buffer[0] = 1;
    for (int i = 0; i < ksize; i++) {
      buffer[i + 1] = 0;
    }
    for (int i = 0; i < ksize - flag - 1; i++) {
      old = buffer[0];
      for (int j = 1; j <= ksize; j++) {
        now = buffer[j] + buffer[j - 1];
        buffer[j - 1] = old;
        old = now;
      }
    }
    for (int i = 0; i < flag; i++) {
      old = -buffer[0];
      for (int j = 1; j <= ksize; j++) {
        now = buffer[j - 1] - buffer[j];
        buffer[j - 1] = old;
        old = now;
      }
    }
  }

  scale = flag == 0 ? scale : 1.0;
  for (int i = 0; i < ksize; i++) {
    kernel[i] = buffer[i] * scale;
  }
}

bool Sobel(const LiteMat &src, LiteMat &dst, int flag_x, int flag_y, int ksize, double scale,  // NOLINT
           PaddBorderType pad_type) {
  if (src.IsEmpty() || src.data_type_ != LDataType::UINT8) {
    return false;
  }
  if (flag_x < 0 || flag_y < 0 || flag_x + flag_y <= 0 || flag_x >= ksize || flag_y >= ksize) {
    return false;
  }

  if (dst.IsEmpty() || dst.width_ != src.width_ || dst.height_ != src.height_ || dst.channel_ != src.channel_ ||
      dst.data_type_ != LDataType::FLOAT32) {
    dst.Init(src.width_, src.height_, src.channel_, LDataType::FLOAT32);
  }

  LiteMat kx, ky;
  kx.Init(ksize, 1, 1, LDataType::FLOAT32);
  ky.Init(1, ksize, 1, LDataType::FLOAT32);

  GetSobelKernel(kx, flag_x, ksize, scale);
  GetSobelKernel(ky, flag_y, ksize, scale);

  return ConvRowCol(src, kx, ky, dst, LDataType::FLOAT32, pad_type);
}

static float GetEdge(const std::vector<float> &temp, int width, int height, int x, int y) {
  if (x >= 0 && y >= 0 && x < width && y < height) {
    return temp[y * width + x];
  } else {
    return -1.0f;
  }
}

static float Round(float value) {
  // rounding if the result is even
  // eg. 1.5 -> 2, 2.5 -> 2
  float rnd = round(value);
  float rnd_l = floor(value);
  float rnd_h = ceil(value);
  if (value - rnd_l == 0.5) {
    if (fmod(rnd, 2) == 0) {
      return rnd;
    } else if (value > 0) {
      return rnd_l;
    } else {
      return rnd_h;
    }
  }
  return rnd;
}

static void NonMaximumSuppression(const LiteMat &gx, const LiteMat &gy, LiteMat &edges, bool L2gradient) {  // NOLINT
  edges.Init(gx.width_, gx.height_, gx.channel_, gx.data_type_);

  const float *gx_ptr = gx;
  const float *gy_ptr = gy;
  float *edges_ptr = edges;

  int size = gx.height_ * gx.width_;
  std::vector<float> temp(size);
  for (int i = 0; i < size; i++) {
    float gx_value = Round(gx_ptr[i]);
    float gy_value = Round(gy_ptr[i]);
    if (L2gradient) {
      temp[i] = sqrt(gx_value * gx_value + gy_value * gy_value);
    } else {
      temp[i] = abs(gx_value) + abs(gy_value);
    }
  }

  for (int y = 0; y < gx.height_; y++) {
    for (int x = 0; x < gx.width_; x++) {
      float gx_value = Round(gx_ptr[y * gx.width_ + x]);
      float gy_value = Round(gy_ptr[y * gx.width_ + x]);

      float gx_value_abs = std::abs(gx_value);
      float gy_value_abs = std::abs(gy_value);
      float angle_value = atan2(gy_value_abs, gx_value_abs);
      float edge_value = temp[y * gx.width_ + x];
      float edge_pre, edge_nex;
      if (angle_value < ANGLE_22_5 || angle_value > ANGLE_67_5) {
        if (angle_value < ANGLE_22_5) {
          edge_pre = GetEdge(temp, gx.width_, gx.height_, x - 1, y);
          edge_nex = GetEdge(temp, gx.width_, gx.height_, x + 1, y);
        } else {
          edge_pre = GetEdge(temp, gx.width_, gx.height_, x, y - 1);
          edge_nex = GetEdge(temp, gx.width_, gx.height_, x, y + 1);
        }
        if (edge_value > edge_pre && edge_value >= edge_nex) {
          edges_ptr[y * gx.width_ + x] = temp[y * gx.width_ + x];
        } else {
          edges_ptr[y * gx.width_ + x] = 0.f;
        }
      } else {
        if (gx_value * gy_value < 0) {
          edge_pre = GetEdge(temp, gx.width_, gx.height_, x + 1, y - 1);
          edge_nex = GetEdge(temp, gx.width_, gx.height_, x - 1, y + 1);
        } else {
          edge_pre = GetEdge(temp, gx.width_, gx.height_, x - 1, y - 1);
          edge_nex = GetEdge(temp, gx.width_, gx.height_, x + 1, y + 1);
        }
        if (edge_value > edge_pre && edge_value > edge_nex) {
          edges_ptr[y * gx.width_ + x] = temp[y * gx.width_ + x];
        } else {
          edges_ptr[y * gx.width_ + x] = 0.f;
        }
      }
    }
  }
}

static void Hysteresis(const LiteMat &edges, uint8_t *dst, double low_thresh, double high_thresh) {
  const float *edges_ptr = edges;

  int size = edges.height_ * edges.width_;
  std::vector<int> stack;
  std::vector<int> buffer(size);
  int buffer_step = edges.width_;
  for (int y = 0; y < edges.height_; y++) {
    for (int x = 0; x < edges.width_; x++) {
      int pos = y * edges.width_ + x;
      float edge_value = edges_ptr[pos];
      if (edge_value > high_thresh) {
        buffer[pos] = 2;
        stack.push_back(pos);
      } else if (edge_value <= low_thresh) {
        buffer[pos] = 0;
      } else {
        buffer[pos] = 1;
      }
    }
  }

  while (!stack.empty()) {
    int pos = stack.back();
    stack.pop_back();
    int y = static_cast<int>(pos / buffer_step);
    int x = pos % buffer_step;
    for (int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        int next_y = y + i;
        int next_x = x + j;
        if (next_y < 0 || next_x < 0 || next_y >= edges.height_ || next_x >= edges.width_ ||
            (next_y == y && next_x == x)) {
          continue;
        }
        int next = next_y * buffer_step + next_x;
        if (buffer[next] == 1) {
          buffer[next] = 2;
          stack.push_back(next);
        }
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (buffer[i] == 2) {
      dst[i] = 255;
    } else {
      dst[i] = 0;
    }
  }
}

bool Canny(const LiteMat &src, LiteMat &dst, double low_thresh, double high_thresh, int ksize,  // NOLINT
           bool L2gradient) {
  if (src.IsEmpty() || src.data_type_ != LDataType::UINT8 || src.channel_ != 1) {
    return false;
  }
  if (low_thresh < 0 || high_thresh < 0 || low_thresh > high_thresh) {
    return false;
  }
  if (ksize % 2 == 0 || ksize < 3 || ksize > 7) {
    return false;
  }
  if (dst.IsEmpty() || dst.width_ != src.width_ || dst.height_ != src.height_ || dst.channel_ != src.channel_ ||
      dst.data_type_ != src.data_type_) {
    dst.Init(src.width_, src.height_, src.channel_, src.data_type_);
  }

  double scale = ksize == 7 ? 1 / 16.0 : 1.0;
  low_thresh *= scale;
  high_thresh *= scale;

  LiteMat gx, gy;
  Sobel(src, gx, 1, 0, ksize, scale, PaddBorderType::PADD_BORDER_REPLICATE);
  Sobel(src, gy, 0, 1, ksize, scale, PaddBorderType::PADD_BORDER_REPLICATE);

  LiteMat edges;
  NonMaximumSuppression(gx, gy, edges, L2gradient);

  Hysteresis(edges, dst, low_thresh, high_thresh);
  return true;
}
}  // namespace dataset
}  // namespace mindspore
