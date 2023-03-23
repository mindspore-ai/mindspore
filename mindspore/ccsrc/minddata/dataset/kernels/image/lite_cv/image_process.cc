/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/lite_cv/image_process.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

namespace mindspore {
namespace dataset {
constexpr uint32_t kR2Gray = 9798;
constexpr uint32_t kG2Gray = 19235;
constexpr uint32_t kB2Gray = 3735;
constexpr int32_t kGrayShift = 15;
constexpr int32_t kGrayShiftDelta = 1 << (kGrayShift - 1);
constexpr int32_t kYScale = 0x0101;
constexpr int32_t kU2B = -128;
constexpr int32_t kU2G = 25;
constexpr int32_t kV2R = -102;
constexpr int32_t kV2G = 52;
constexpr int32_t kY2G = 18997;
constexpr int32_t kY2GB = -1160;
constexpr int32_t kB2B = kU2B * 128 + kY2GB;
constexpr int32_t kB2G = kU2G * 128 + kV2G * 128 + kY2GB;
constexpr int32_t kB2R = kV2R * 128 + kY2GB;

static bool Equal(const float &a, const float &b) { return std::fabs(a - b) < 1e-6; }

static inline bool InitBilinearWeight(int *data_ptr, int16_t *weight_ptr, double scale, int dst_length, int src_length,
                                      int a) {
  const int RESIZE_SCALE = 1 << 11;
  if (data_ptr == nullptr || weight_ptr == nullptr) {
    return false;
  }

  int *data_start_ptr = data_ptr;
  int16_t *weight_start_ptr = weight_ptr;

  for (unsigned int i = 0; i < dst_length; i++) {
    float src_f_x = static_cast<float>((i + 0.5) * scale - 0.5);
    int src_u_x = static_cast<int>(floor(src_f_x));
    src_f_x -= src_u_x;
    if (src_u_x < 0) {
      src_u_x = 0;
      src_f_x = 0.0f;
    }
    if (src_u_x >= src_length - 1) {
      src_u_x = src_length - 2;
      src_f_x = 1.0f;
    }
    data_start_ptr[i] = src_u_x * a;
    int16_t t0 = INT16_CAST((1.0f - src_f_x) * RESIZE_SCALE);
    int16_t t1 = INT16_CAST(src_f_x * RESIZE_SCALE);

    weight_start_ptr[i * 2] = t0;
    weight_start_ptr[i * 2 + 1] = t1;
  }
  return true;
}

static bool ResizeBilinear3C(const unsigned char *src, int src_width, int src_height, unsigned char *dst, int dst_width,
                             int dst_height) {
  double scale_width = static_cast<double>(src_width) / dst_width;
  double scale_height = static_cast<double>(src_height) / dst_height;

  if (dst_height >= (INT_MAX / 2 - dst_width)) {
    return false;
  }
  if (dst_height >= (INT_MAX / 3 / dst_width)) {
    return false;
  }

  // The allocate memory cannot exceed 2GB.
  if ((2 * sizeof(int) * (2 * dst_width + dst_height)) > INT_MAX) {
    return false;
  }
  int *data_buf = new (std::nothrow) int[2 * sizeof(int) * (2 * dst_width + dst_height)];
  if (data_buf == nullptr) {
    return false;
  }

  int *x_offset = data_buf;
  int *y_offset = data_buf + dst_width + dst_width;

  int16_t *x_weight = reinterpret_cast<int16_t *>(data_buf + dst_width);
  int16_t *y_weight = reinterpret_cast<int16_t *>(data_buf + dst_width + dst_width + dst_height);

  if (!InitBilinearWeight(x_offset, x_weight, scale_width, dst_width, src_width, 3)) {
    delete[] data_buf;
    return false;
  }
  if (!InitBilinearWeight(y_offset, y_weight, scale_height, dst_height, src_height, 1)) {
    delete[] data_buf;
    return false;
  }

  LiteMat x_tmp_buf0(dst_width * 3 + 1, LDataType::UINT16);
  LiteMat x_tmp_buf1(dst_width * 3 + 1, LDataType::UINT16);
  int16_t *row0_ptr = reinterpret_cast<int16_t *>(x_tmp_buf0.data_ptr_);
  int16_t *row1_ptr = reinterpret_cast<int16_t *>(x_tmp_buf1.data_ptr_);

  int prev_height = -2;

  for (int y = 0; y < dst_height; y++) {
    int y_span = y_offset[y];

    if (y_span == prev_height) {
    } else if (y_span == prev_height + 1) {
      int16_t *tmp = row0_ptr;
      row0_ptr = row1_ptr;
      row1_ptr = tmp;
      const unsigned char *src_start = src + 3 * src_width * (y_span + 1);
      const int16_t *x_weight_p = x_weight;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src_start_p = src_start + x_offset[x];
        row1_ptr1[0] = (src_start_p[0] * x_weight_p[0] + src_start_p[3] * x_weight_p[1]) >> 4;
        row1_ptr1[1] = (src_start_p[1] * x_weight_p[0] + src_start_p[4] * x_weight_p[1]) >> 4;
        row1_ptr1[2] = (src_start_p[2] * x_weight_p[0] + src_start_p[5] * x_weight_p[1]) >> 4;
        x_weight_p += 2;
        row1_ptr1 += 3;
      }
    } else {
      const unsigned char *src0 = src + 3 * src_width * (y_span);
      const unsigned char *src1 = src + 3 * src_width * (y_span + 1);

      const int16_t *x_weight_ptr = x_weight;
      int16_t *row0_ptr0 = row0_ptr;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src0_ptr = src0 + x_offset[x];
        const unsigned char *src1_ptr = src1 + x_offset[x];

        for (int c = 0; c < 3; c++) {
          row0_ptr0[c] = (src0_ptr[c] * x_weight_ptr[0] + src0_ptr[c + 3] * x_weight_ptr[1]) >> 4;
          row1_ptr1[c] = (src1_ptr[c] * x_weight_ptr[0] + src1_ptr[c + 3] * x_weight_ptr[1]) >> 4;
        }

        x_weight_ptr += 2;
        row0_ptr0 += 3;
        row1_ptr1 += 3;
      }
    }
    prev_height = y_span;

    int16_t *row0_ptr0 = row0_ptr;
    int16_t *row1_ptr1 = row1_ptr;
    unsigned char *dst_ptr = dst + dst_width * 3 * (y);

    for (int k = 0; k < dst_width * 3; k++) {
      int16_t t0 = (int16_t)((y_weight[0] * (int16_t)(*row0_ptr0++)) >> 16);
      int16_t t1 = (int16_t)((y_weight[1] * (int16_t)(*row1_ptr1++)) >> 16);
      *dst_ptr++ = static_cast<unsigned char>((t0 + t1 + 2) >> 2);
    }
    y_weight += 2;
  }
  delete[] data_buf;
  return true;
}

static bool ResizeBilinear1C(const unsigned char *src, int src_width, int src_height, unsigned char *dst, int dst_width,
                             int dst_height) {
  double scale_width = static_cast<double>(src_width) / dst_width;
  double scale_height = static_cast<double>(src_height) / dst_height;

  if (dst_height >= (INT_MAX / 2 - dst_width)) {
    return false;
  }
  if (dst_height >= (INT_MAX / dst_width)) {
    return false;
  }

  // The allocate memory cannot exceed 2GB.
  if ((2 * sizeof(int) * (2 * dst_width + dst_height)) > INT_MAX) {
    return false;
  }
  int *data_buf = new (std::nothrow) int[2 * sizeof(int) * (2 * dst_width + dst_height)];
  if (data_buf == nullptr) {
    return false;
  }

  int *x_offset = data_buf;
  int *y_offset = data_buf + dst_width + dst_width;

  int16_t *x_weight = reinterpret_cast<int16_t *>(data_buf + dst_width);
  int16_t *y_weight = reinterpret_cast<int16_t *>(data_buf + dst_width + dst_width + dst_height);

  if (!InitBilinearWeight(x_offset, x_weight, scale_width, dst_width, src_width, 1)) {
    delete[] data_buf;
    return false;
  }
  if (!InitBilinearWeight(y_offset, y_weight, scale_height, dst_height, src_height, 1)) {
    delete[] data_buf;
    return false;
  }

  LiteMat x_tmp_buf0(dst_width, LDataType::UINT16);
  LiteMat x_tmp_buf1(dst_width, LDataType::UINT16);
  int16_t *row0_ptr = reinterpret_cast<int16_t *>(x_tmp_buf0.data_ptr_);
  int16_t *row1_ptr = reinterpret_cast<int16_t *>(x_tmp_buf1.data_ptr_);

  int prev_height = -2;

  for (int y = 0; y < dst_height; y++) {
    int y_span = y_offset[y];

    if (y_span == prev_height) {
    } else if (y_span == prev_height + 1) {
      int16_t *tmp = row0_ptr;
      row0_ptr = row1_ptr;
      row1_ptr = tmp;
      const unsigned char *src_start = src + src_width * (y_span + 1);
      const int16_t *x_weight_p = x_weight;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src_start_p = src_start + x_offset[x];
        if ((src_start_p + 3 - src) >= (src_width * src_height)) {
          continue;
        }
        row1_ptr1[x] = (src_start_p[0] * x_weight_p[0] + src_start_p[3] * x_weight_p[1]) >> 4;
        x_weight_p += 2;
      }
    } else {
      const unsigned char *src0 = src + src_width * (y_span);
      const unsigned char *src1 = src + src_width * (y_span + 1);

      const int16_t *x_weight_ptr = x_weight;
      int16_t *row0_ptr0 = row0_ptr;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src0_ptr = src0 + x_offset[x];
        const unsigned char *src1_ptr = src1 + x_offset[x];

        row0_ptr0[x] = (src0_ptr[0] * x_weight_ptr[0] + src0_ptr[3] * x_weight_ptr[1]) >> 4;
        row1_ptr1[x] = (src1_ptr[0] * x_weight_ptr[0] + src1_ptr[3] * x_weight_ptr[1]) >> 4;

        x_weight_ptr += 2;
      }
    }
    prev_height = y_span;

    int16_t *row0_ptr0 = row0_ptr;
    int16_t *row1_ptr1 = row1_ptr;
    unsigned char *dst_ptr = dst + dst_width * (y);

    for (int k = 0; k < dst_width; k++) {
      int16_t t0 = (int16_t)((y_weight[0] * (int16_t)(*row0_ptr0++)) >> 16);
      int16_t t1 = (int16_t)((y_weight[1] * (int16_t)(*row1_ptr1++)) >> 16);
      *dst_ptr++ = static_cast<unsigned char>((t0 + t1 + 2) >> 2);
    }

    y_weight += 2;
  }
  delete[] data_buf;
  return true;
}

static inline uint8_t clip(float value) {
  int int_val = roundf(value);
  return std::max<int32_t>(std::numeric_limits<uint8_t>::min(),
                           std::min<int32_t>(std::numeric_limits<uint8_t>::max(), int_val));
}

template <typename T1, typename T2>
static bool Conv2DImplement(const LiteMat &src, const LiteMat &kernel, T2 *dst, LDataType dst_type,
                            PaddBorderType pad_type) {
  int border_x = static_cast<int>(kernel.width_ / 2);
  int border_y = static_cast<int>(kernel.height_ / 2);

  LiteMat pad_mat;

  if ((border_x > INT_MAX / 2) || (src.width_ > INT_MAX - 2 * border_x)) {
    return false;
  }
  if ((border_y > INT_MAX / 2) || (src.height_ > INT_MAX - 2 * border_y)) {
    return false;
  }

  pad_mat.Init(src.width_ + 2 * border_x, src.height_ + 2 * border_y, src.channel_, src.data_type_);
  RETURN_FALSE_IF_LITEMAT_EMPTY(pad_mat);

  if (!Pad(src, pad_mat, border_y, border_y, border_x, border_x, pad_type)) {
    return false;
  }

  const T1 *pad_ptr = pad_mat;
  const float *kernel_ptr = kernel;

  int pad_step = pad_mat.width_ * pad_mat.channel_;
  int dst_step = src.width_ * src.channel_;

  if (src.channel_ == 1) {
    for (int y = border_y; y < pad_mat.height_ - border_y; y++) {
      for (int x = border_x; x < pad_mat.width_ - border_x; x++) {
        float conv_sum = 0;
        for (int i = -border_y; i < -border_y + kernel.height_; i++) {
          for (int j = -border_x; j < -border_x + kernel.width_; j++) {
            conv_sum += pad_ptr[(y + i) * pad_step + (x + j) * pad_mat.channel_] *
                        kernel_ptr[(i + border_y) * kernel.width_ + (j + border_x)];
          }
        }
        if (dst_type == LDataType::UINT8) {
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_] = clip(conv_sum);
        } else {
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_] = conv_sum;
        }
      }
    }
  } else if (src.channel_ == 3) {
    for (int y = border_y; y < pad_mat.height_ - border_y; y++) {
      for (int x = border_x; x < pad_mat.width_ - border_x; x++) {
        float conv_sum_b = 0;
        float conv_sum_g = 0;
        float conv_sum_r = 0;
        for (int i = -border_y; i < -border_y + kernel.height_; i++) {
          for (int j = -border_x; j < -border_x + kernel.width_; j++) {
            conv_sum_b += pad_ptr[(y + i) * pad_step + (x + j) * pad_mat.channel_] *
                          kernel_ptr[(i + border_y) * kernel.width_ + (j + border_x)];
            conv_sum_g += pad_ptr[(y + i) * pad_step + (x + j) * pad_mat.channel_ + 1] *
                          kernel_ptr[(i + border_y) * kernel.width_ + (j + border_x)];
            conv_sum_r += pad_ptr[(y + i) * pad_step + (x + j) * pad_mat.channel_ + 2] *
                          kernel_ptr[(i + border_y) * kernel.width_ + (j + border_x)];
          }
        }
        if (dst_type == LDataType::UINT8) {
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_] = clip(conv_sum_b);
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_ + 1] = clip(conv_sum_g);
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_ + 2] = clip(conv_sum_r);
        } else {
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_] = conv_sum_b;
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_ + 1] = conv_sum_g;
          dst[(y - border_y) * dst_step + (x - border_x) * src.channel_ + 2] = conv_sum_r;
        }
      }
    }
  } else {
    return false;
  }
  return true;
}

bool Conv2D(const LiteMat &src, const LiteMat &kernel, LiteMat &dst, LDataType dst_type, PaddBorderType pad_type) {
  if (src.IsEmpty() || kernel.IsEmpty()) {
    return false;
  }
  if ((dst_type != LDataType::UINT8 && dst_type != LDataType::FLOAT32) || kernel.data_type_ != LDataType::FLOAT32) {
    return false;
  }
  if (dst.IsEmpty() || dst.width_ != src.width_ || dst.height_ != src.height_ || dst.channel_ != src.channel_ ||
      dst.data_type_ != dst_type) {
    dst.Init(src.width_, src.height_, src.channel_, dst_type);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  }

  if (src.data_type_ == LDataType::UINT8 && dst.data_type_ == LDataType::UINT8) {
    return Conv2DImplement<uint8_t, uint8_t>(src, kernel, dst, dst_type, pad_type);
  } else if (src.data_type_ == LDataType::UINT8 && dst.data_type_ == LDataType::FLOAT32) {
    return Conv2DImplement<uint8_t, float>(src, kernel, dst, dst_type, pad_type);
  } else if (src.data_type_ == LDataType::FLOAT32 && dst.data_type_ == LDataType::UINT8) {
    return Conv2DImplement<float, uint8_t>(src, kernel, dst, dst_type, pad_type);
  } else if (src.data_type_ == LDataType::FLOAT32 && dst.data_type_ == LDataType::FLOAT32) {
    return Conv2DImplement<float, float>(src, kernel, dst, dst_type, pad_type);
  } else {
    return false;
  }
}

bool ConvRowCol(const LiteMat &src, const LiteMat &kx, const LiteMat &ky, LiteMat &dst, LDataType dst_type,
                PaddBorderType pad_type) {
  if (src.IsEmpty() || kx.IsEmpty() || ky.IsEmpty()) {
    return false;
  }
  if (dst_type != LDataType::UINT8 && dst_type != LDataType::FLOAT32) {
    return false;
  }
  if (dst.IsEmpty() || dst.width_ != src.width_ || dst.height_ != src.height_ || dst.channel_ != src.channel_ ||
      dst.data_type_ != dst_type) {
    dst.Init(src.width_, src.height_, src.channel_, dst_type);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  }

  LiteMat mid;
  bool ret = Conv2D(src, kx, mid, LDataType::FLOAT32, pad_type) && Conv2D(mid, ky, dst, dst_type, pad_type);
  return ret;
}

bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h) {
  if (dst_h <= 0 || dst_w <= 0) {
    return false;
  }
  if (src.data_type_ != LDataType::UINT8) {
    return false;
  }
  if (src.channel_ != 3 && src.channel_ != 1) {
    return false;
  }
  if (dst.IsEmpty()) {
    dst.Init(dst_w, dst_h, src.channel_, LDataType::UINT8);
    if (dst.IsEmpty()) {
      return false;
    }
  } else if (dst.height_ != dst_h || dst.width_ != dst_w || dst.channel_ != src.channel_) {
    return false;
  } else if (dst.data_type_ != LDataType::UINT8) {
    return false;
  } else {
  }

  if (src.channel_ == 3) {
    const unsigned char *src_start_p = src;
    unsigned char *dst_start_p = dst;
    return ResizeBilinear3C(src_start_p, src.width_, src.height_, dst_start_p, dst_w, dst_h);
  } else {  // channel == 1
    const unsigned char *src_start_p = src;
    unsigned char *dst_start_p = dst;
    return ResizeBilinear1C(src_start_p, src.width_, src.height_, dst_start_p, dst_w, dst_h);
  }
}

static bool ConvertBGR(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    unsigned char *dst_ptr = mat;
    // mindspore lite version, there is no securec lib
    (void)memcpy(dst_ptr, data, w * h * 3 * sizeof(unsigned char));
  } else {
    return false;
  }
  return true;
}

static bool ConvertRGBAToBGR(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[0] = data_ptr[2];
        ptr[1] = data_ptr[1];
        ptr[2] = data_ptr[0];
        ptr += 3;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

static bool ConvertRGBAToRGB(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[0] = data_ptr[0];
        ptr[1] = data_ptr[1];
        ptr[2] = data_ptr[2];
        ptr += 3;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

static bool ConvertYUV420SPToBGR(const uint8_t *data, LDataType data_type, bool flag, int w, int h, LiteMat &mat) {
  if (data == nullptr || w <= 0 || h <= 0) {
    return false;
  }
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    const uint8_t *y_ptr = data;
    const uint8_t *uv_ptr = y_ptr + w * h;
    uint8_t *bgr_ptr = mat;
    const int bgr_stride = 3 * w;

    for (uint64_t y = 0; y < h; ++y) {
      uint8_t *bgr_buf = bgr_ptr;
      const uint8_t *uv_buf = uv_ptr;
      const uint8_t *y_buf = y_ptr;
      uint8_t u;
      uint8_t v;
      for (int x = 0; x < w - 1; x += 2) {
        if (flag) {
          // NV21
          u = uv_buf[1];
          v = uv_buf[0];
        } else {
          // NV12
          u = uv_buf[0];
          v = uv_buf[1];
        }
        uint32_t tmp_y = (uint32_t)(y_buf[0] * kYScale * kY2G) >> 16;
        // b
        bgr_buf[0] = std::clamp((int32_t)(-(u * kU2B) + tmp_y + kB2B) >> 6, 0, 255);
        // g
        bgr_buf[1] = std::clamp((int32_t)(-(u * kU2G + v * kV2G) + tmp_y + kB2G) >> 6, 0, 255);
        // r
        bgr_buf[2] = std::clamp((int32_t)(-(v * kV2R) + tmp_y + kB2R) >> 6, 0, 255);

        tmp_y = (uint32_t)(y_buf[1] * kYScale * kY2G) >> 16;
        bgr_buf[3] = std::clamp((int32_t)(-(u * kU2B) + tmp_y + kB2B) >> 6, 0, 255);
        bgr_buf[4] = std::clamp((int32_t)(-(u * kU2G + v * kV2G) + tmp_y + kB2G) >> 6, 0, 255);
        bgr_buf[5] = std::clamp((int32_t)(-(v * kV2R) + tmp_y + kB2R) >> 6, 0, 255);

        y_buf += 2;
        uv_buf += 2;
        bgr_buf += 6;
      }
      if (w & 1) {
        if (flag) {
          // NV21
          u = uv_buf[1];
          v = uv_buf[0];
        } else {
          // NV12
          u = uv_buf[0];
          v = uv_buf[1];
        }
        uint32_t tmp_y = (uint32_t)(y_buf[0] * kYScale * kY2G) >> 16;
        bgr_buf[0] = std::clamp((int32_t)(-(u * kU2B) + tmp_y + kB2B) >> 6, 0, 255);
        bgr_buf[1] = std::clamp((int32_t)(-(u * kU2G + v * kV2G) + tmp_y + kB2G) >> 6, 0, 255);
        bgr_buf[2] = std::clamp((int32_t)(-(v * kV2R) + tmp_y + kB2R) >> 6, 0, 255);
      }

      bgr_ptr += bgr_stride;
      y_ptr += w;
      if (y & 1) {
        uv_ptr += w;
      }
    }
  }
  return true;
}

static bool ConvertRGBAToGRAY(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 1, LDataType::UINT8);
    if (mat.IsEmpty()) {
      return false;
    }
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        *ptr = (data_ptr[2] * kB2Gray + data_ptr[1] * kG2Gray + data_ptr[0] * kR2Gray + kGrayShiftDelta) >> kGrayShift;
        ptr++;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m) {
  if (data == nullptr) {
    return false;
  }
  if (w <= 0 || h <= 0) {
    return false;
  }

  if (data_type != LDataType::UINT8) {
    return false;
  }
  if (pixel_type == LPixelType::RGBA2BGR) {
    return ConvertRGBAToBGR(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGBA2GRAY) {
    return ConvertRGBAToGRAY(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGBA2RGB) {
    return ConvertRGBAToRGB(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::NV212BGR) {
    return ConvertYUV420SPToBGR(data, data_type, true, w, h, m);
  } else if (pixel_type == LPixelType::NV122BGR) {
    return ConvertYUV420SPToBGR(data, data_type, false, w, h, m);
  } else if (pixel_type == LPixelType::BGR) {
    return ConvertBGR(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGB) {
    return ConvertBGR(data, data_type, w, h, m);
  } else {
    return false;
  }
}

bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale) {
  if (src.data_type_ != LDataType::UINT8) {
    return false;
  }

  if (scale < 0.0 || scale > 100) {
    return false;
  }

  if (dst.IsEmpty()) {
    dst.Init(src.width_, src.height_, src.channel_, LDataType::FLOAT32);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (src.width_ != dst.width_ || src.height_ != dst.height_ || src.channel_ != dst.channel_ ||
             dst.data_type_ != LDataType::FLOAT32) {
    return false;
  }

  const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(src.data_ptr_);
  float *dst_ptr = reinterpret_cast<float *>(dst.data_ptr_);
  int64_t total_size = src.height_ * src.width_ * src.channel_;
  int64_t x = 0;
#ifdef ENABLE_NEON
  float32x4_t v_scale = vdupq_n_f32(static_cast<float>(scale));
  float32x4_t v_c = vdupq_n_f32(0.0f);
  const int64_t step = 16;
  for (; x <= total_size - step; x += step) {
    uint8x16_t v_src = vld1q_u8(src_ptr + x);
    uint8x16_t v_dst;

    uint16x8_t v_l_16x8 = vmovl_u8(vget_low_u8(v_src));
    uint16x8_t v_h_16x8 = vmovl_u8(vget_high_u8(v_src));

    float32x4_t v_ll_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_l_16x8)));
    float32x4_t v_lh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_l_16x8)));
    float32x4_t v_hl_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_h_16x8)));
    float32x4_t v_hh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_h_16x8)));

#if defined(__aarch64__) || defined(_M_ARM64)
    v_ll_f32x4 = vfmaq_f32(v_c, v_ll_f32x4, v_scale);
    v_lh_f32x4 = vfmaq_f32(v_c, v_lh_f32x4, v_scale);
    v_hl_f32x4 = vfmaq_f32(v_c, v_hl_f32x4, v_scale);
    v_hh_f32x4 = vfmaq_f32(v_c, v_hh_f32x4, v_scale);
#else
    v_ll_f32x4 = vmlaq_f32(v_c, v_ll_f32x4, v_scale);
    v_lh_f32x4 = vmlaq_f32(v_c, v_lh_f32x4, v_scale);
    v_hl_f32x4 = vmlaq_f32(v_c, v_hl_f32x4, v_scale);
    v_hh_f32x4 = vmlaq_f32(v_c, v_hh_f32x4, v_scale);
#endif

    vst1q_f32(dst_ptr + x, v_ll_f32x4);
    vst1q_f32(dst_ptr + x + 4, v_lh_f32x4);
    vst1q_f32(dst_ptr + x + 8, v_hl_f32x4);
    vst1q_f32(dst_ptr + x + 12, v_hh_f32x4);
  }
#endif
  for (; x < total_size; x++) {
    dst_ptr[x] = static_cast<float>(src_ptr[x] * scale);
  }
  return true;
}

template <typename T>
static bool CropInternal(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h) {
  int dst_h = h;
  int dst_w = w;
  int dst_c = src.channel_;
  if (dst.IsEmpty()) {
    dst.Init(dst_w, dst_h, dst_c, src.data_type_);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (dst.height_ != h || dst.width_ != w || dst.channel_ != src.channel_) {
    return false;
  } else if (dst.data_type_ != src.data_type_) {
    return false;
  }
  const T *src_start_p = src;
  T *dst_start_p = dst;
  for (int i_h = 0; i_h < dst_h; i_h++) {
    const T *src_index_p = src_start_p + (y + i_h) * src.width_ * dst_c + x * dst_c;
    T *dst_index_p = dst_start_p + i_h * dst_w * dst_c;
    // mindspore lite version, there is no securec lib
    (void)memcpy(dst_index_p, src_index_p, dst_w * dst_c * sizeof(T));
  }
  return true;
}

bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h) {
  if (x < 0 || y < 0 || w <= 0 || h <= 0) {
    return false;
  }
  if (y > src.height_ - h || x > src.width_ - w) {
    return false;
  }

  if (src.data_type_ == LDataType::UINT8) {
    return CropInternal<uint8_t>(src, dst, x, y, w, h);
  } else if (src.data_type_ == LDataType::FLOAT32) {
    return CropInternal<float>(src, dst, x, y, w, h);
  } else {
    return false;
  }
}

static bool CheckZero(const std::vector<float> &vs) {
  return std::any_of(vs.begin(), vs.end(), [](const float &v) { return Equal(v, 0.0f); });
}

static bool CheckZero(const std::vector<size_t> &vs) {
  return std::any_of(vs.begin(), vs.end(), [](const float &v) { return v == 0; });
}

static bool CheckMeanAndStd(const LiteMat &src, LiteMat &dst, int channel, const std::vector<float> &mean,
                            const std::vector<float> &std) {
  if (mean.empty() && std.empty()) {
    return false;
  }
  if (src.data_type_ != LDataType::FLOAT32 && src.data_type_ != LDataType::UINT8) {
    return false;
  }
  if (mean.size() > 0) {
    if (mean.size() != channel) {
      return false;
    }
  }
  if (std.size() > 0) {
    if (CheckZero(std)) {
      return false;
    }
    if (std.size() != channel) {
      return false;
    }
  }
  if (dst.IsEmpty()) {
    dst.Init(src.width_, src.height_, src.channel_, LDataType::FLOAT32);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (dst.height_ != src.height_ || dst.width_ != src.width_ || dst.channel_ != src.channel_) {
    return false;
  } else if (dst.data_type_ != LDataType::FLOAT32) {
    return false;
  }
  return true;
}

bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean,
                            const std::vector<float> &std) {
  if (!CheckMeanAndStd(src, dst, src.channel_, mean, std)) {
    return false;
  }
  LiteMat src_f;
  if (src.data_type_ == LDataType::UINT8) {
    ConvertTo(src, src_f, 1.0);
  } else {
    src_f = src;
  }

  const float *src_start_p = src_f;
  float *dst_start_p = dst;
  if ((!mean.empty()) && std.empty()) {
    for (int h = 0; h < src_f.height_; h++) {
      for (int w = 0; w < src_f.width_; w++) {
        uint32_t src_start = (h * src_f.width_ + w) * src_f.channel_;
        for (int c = 0; c < src_f.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = src_start_p[index] - mean[c];
        }
      }
    }
  } else if (mean.empty() && (!std.empty())) {
    for (int h = 0; h < src_f.height_; h++) {
      for (int w = 0; w < src_f.width_; w++) {
        uint32_t src_start = (h * src_f.width_ + w) * src_f.channel_;
        for (int c = 0; c < src_f.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = src_start_p[index] / std[c];
        }
      }
    }
  } else if ((!mean.empty()) && (!std.empty())) {
    for (int h = 0; h < src_f.height_; h++) {
      for (int w = 0; w < src_f.width_; w++) {
        uint32_t src_start = (h * src_f.width_ + w) * src_f.channel_;
        for (int c = 0; c < src_f.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = (src_start_p[index] - mean[c]) / std[c];
        }
      }
    }
  } else {
    return false;
  }
  return true;
}

template <typename T>
static void PadWithConstant(const LiteMat &src, LiteMat &dst, const int top, const int bottom, const int left,
                            const int right, const PaddBorderType pad_type, uint8_t fill_b_or_gray, uint8_t fill_g,
                            uint8_t fill_r) {
  std::vector<uint8_t> row_buffer(dst.width_ * dst.channel_ * dst.elem_size_);
  T *const_ptr = reinterpret_cast<T *>(row_buffer.data());
  int src_step = src.width_ * src.channel_ * src.elem_size_;
  int dst_step = dst.width_ * dst.channel_ * dst.elem_size_;
  if (dst.channel_ == 1) {
    for (int i = 0; i < dst_step; i++) {
      const_ptr[i] = fill_b_or_gray;
    }
  } else if (dst.channel_ == 3) {
    for (int i = 0; i < dst.width_; i++) {
      const_ptr[i * dst.channel_] = fill_b_or_gray;
      const_ptr[i * dst.channel_ + 1] = fill_g;
      const_ptr[i * dst.channel_ + 2] = fill_r;
    }
  }

  uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst.data_ptr_);
  uint8_t *src_ptr = reinterpret_cast<uint8_t *>(src.data_ptr_);
  for (int i = 0; i < top; i++) {
    // mindspore lite version, there is no securec lib
    memcpy(dst_ptr + i * dst_step, const_ptr, dst_step);
  }

  int left_size = left * dst.channel_ * dst.elem_size_;
  int right_size = right * dst.channel_ * dst.elem_size_;
  uint8_t *dst_raw_data = dst_ptr + top * dst_step + left_size;
  for (int i = 0; i < src.height_; i++, dst_raw_data += dst_step, src_ptr += src_step) {
    // mindspore lite version, there is no securec lib
    memcpy(dst_raw_data, src_ptr, src_step);
    memcpy(dst_raw_data - left_size, const_ptr, left_size);
    memcpy(dst_raw_data + src_step, const_ptr, right_size);
  }

  for (int i = dst.height_ - bottom; i < dst.height_; i++) {
    // mindspore lite version, there is no securec lib
    memcpy(dst_ptr + i * dst_step, const_ptr, dst_step);
  }
}

static int PadFromPos(int p, int len, PaddBorderType pad_type) {
  constexpr auto pixel_factor = 2;
  if (p >= 0 && p < len) {
    return p;
  }
  if (pad_type == PaddBorderType::PADD_BORDER_REPLICATE) {
    return p < 0 ? 0 : len - 1;
  } else {
    // calculate the position of pixel in reflect mode like edcb|abcdef|edcb
    while (p < 0 || p >= len) {
      if (p < 0) {
        p = -p;
      } else {
        p = pixel_factor * len - p - pixel_factor;
      }
    }
    return p;
  }
}

template <typename T>
static void PadImplement(const LiteMat &src, LiteMat &dst, const int top, const int bottom, const int left,
                         const int right, const PaddBorderType pad_type) {
  int src_step = src.width_ * src.channel_;
  int dst_step = dst.width_ * dst.channel_;

  uint8_t *src_data_ptr = reinterpret_cast<uint8_t *>(src.data_ptr_);
  uint8_t *dst_data_ptr = reinterpret_cast<uint8_t *>(dst.data_ptr_);
  for (int i = 0; i < src.height_; i++) {
    // mindspore lite version, there is no securec lib
    memcpy(dst_data_ptr + (i + top) * dst.steps_[0] + left * dst.steps_[1], src_data_ptr + i * src.steps_[0],
           src.steps_[0]);
  }

  const T *src_ptr = src;
  T *dst_ptr = dst;
  for (int y = 0; y < dst.height_; y++) {
    for (int x = 0; x < dst.width_; x++) {
      if (y < top || y >= dst.height_ - bottom || x < left || x >= dst.width_ - right) {
        int src_y = PadFromPos(y - top, src.height_, pad_type);
        int src_x = PadFromPos(x - left, src.width_, pad_type);
        for (int cn = 0; cn < dst.channel_; cn++) {
          dst_ptr[y * dst_step + x * dst.channel_ + cn] = src_ptr[src_y * src_step + src_x * src.channel_ + cn];
        }
      }
    }
  }
}

template <typename T>
void ExtractChannelImpl(const T *src_ptr, T *dst_ptr, int height, int width, int channel, int col) {
  int total = height * width;
  int i = 0;
  int src_idx = col;
  for (; i < total; i++, src_idx += channel) {
    dst_ptr[i] = src_ptr[src_idx];
  }
}

bool ExtractChannel(LiteMat &src, LiteMat &dst, int col) {
  if (src.IsEmpty() || col < 0 || col > src.channel_ - 1) {
    return false;
  }

  if (src.data_type_ == LDataType::FLOAT32 || src.data_type_ == LDataType::UINT8) {
    if (dst.IsEmpty() || dst.width_ != src.width_ || dst.height_ != src.height_ || dst.channel_ != 1 ||
        dst.data_type_ != src.data_type_) {
      dst.Init(src.width_, src.height_, 1, src.data_type_);
      RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
    }
  }

  if (src.data_type_ == LDataType::FLOAT32) {
    ExtractChannelImpl<float>(src, dst, src.height_, src.width_, src.channel_, col);
  } else if (src.data_type_ == LDataType::UINT8) {
    ExtractChannelImpl<uint8_t>(src, dst, src.height_, src.width_, src.channel_, col);
  } else {
    return false;
  }
  return true;
}

bool Split(const LiteMat &src, std::vector<LiteMat> &mv) {
  if (src.data_type_ == LDataType::FLOAT32) {
    const float *src_start_p = src;
    for (int c = 0; c < src.channel_; c++) {
      LiteMat dst;
      (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
      RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
      float *dst_start_p = dst;
      for (int h = 0; h < src.height_; h++) {
        uint32_t src_start = h * src.width_ * src.channel_;
        uint32_t dst_start = h * dst.width_;
        for (int w = 0; w < src.width_; w++) {
          uint32_t src_index = src_start + w * src.channel_ + c;
          uint32_t dst_index = dst_start + w;
          dst_start_p[dst_index] = src_start_p[src_index];
        }
      }
      mv.emplace_back(dst);
    }
    return true;
  } else if (src.data_type_ == LDataType::UINT8) {
    const uint8_t *src_start_p = src;
    for (int c = 0; c < src.channel_; c++) {
      LiteMat dst;
      (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
      RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
      uint8_t *dst_start_p = dst;
      for (int h = 0; h < src.height_; h++) {
        uint32_t src_start = h * src.width_ * src.channel_;
        uint32_t dst_start = h * dst.width_;
        for (int w = 0; w < src.width_; w++) {
          uint32_t src_index = src_start + w * src.channel_ + c;
          uint32_t dst_index = dst_start + w;
          dst_start_p[dst_index] = src_start_p[src_index];
        }
      }
      mv.emplace_back(dst);
    }
    return true;
  } else {
    return false;
  }
}

template <typename T>
inline void MergeImpl(const std::vector<LiteMat> &mv, T *dst_ptr, int height, int width, int channel) {
  T *mv_ptr[4];
  int area = height * width;
  for (int c = 0; c < channel; c++) {
    mv_ptr[c] = reinterpret_cast<T *>(mv[c].data_ptr_);
  }
  for (int i = 0; i < area; i++) {
    for (int c = 0; c < channel; c++) {
      dst_ptr[c] = *mv_ptr[c];
      mv_ptr[c]++;
    }
    dst_ptr += channel;
  }
}

bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst) {
  constexpr int64_t mv_size_three = 3;
  constexpr int64_t mv_size_four = 4;
  if (mv.size() != 1 && mv.size() != mv_size_three && mv.size() != mv_size_four) {
    return false;
  }

  int width = mv[0].width_;
  int height = mv[0].height_;
  int channel = mv.size();
  LDataType data_type = mv[0].data_type_;

  // The arrays in list must be single-channel
  if (std::any_of(mv.begin(), mv.end(), [](const LiteMat &m) { return m.channel_ != 1; })) {
    return false;
  }

  for (int i = 1; i < mv.size(); i++) {
    if (width != mv[i].width_ || height != mv[i].height_ || data_type != mv[i].data_type_) {
      return false;
    }
  }

  if (dst.IsEmpty() || dst.width_ != width || dst.height_ != height || dst.channel_ != channel ||
      dst.data_type_ != data_type) {
    dst.Init(width, height, channel, data_type);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  }

  if (dst.data_type_ == LDataType::FLOAT32) {
    MergeImpl<float>(mv, dst, height, width, channel);
  } else if (dst.data_type_ == LDataType::UINT8) {
    MergeImpl<uint8_t>(mv, dst, height, width, channel);
  } else {
    return false;
  }
  return true;
}

inline bool CheckInt(const std::vector<int> &nums) {
  if (std::any_of(nums.begin(), nums.end(), [](const auto &num) { return num < 0; })) {
    return false;
  }
  return true;
}

bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type,
         uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r) {
  RETURN_FALSE_IF_LITEMAT_EMPTY(src);
  if (!CheckInt({top, bottom, left, right})) {
    return false;
  }
  if (src.width_ > std::numeric_limits<int>::max() - left ||
      src.width_ + left > std::numeric_limits<int>::max() - right) {
    return false;
  }
  if (src.height_ > std::numeric_limits<int>::max() - top ||
      src.height_ + top > std::numeric_limits<int>::max() - bottom) {
    return false;
  }
  int dst_width = src.width_ + left + right;
  int dst_height = src.height_ + top + bottom;
  if (dst.IsEmpty()) {
    dst.Init(dst_width, dst_height, src.channel_, src.data_type_);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (dst.width_ != dst_width || dst.height_ != dst_height || src.channel_ != dst.channel_) {
    return false;
  } else if (src.data_type_ != dst.data_type_) {
    return false;
  }
  if (pad_type == PADD_BORDER_CONSTANT && src.data_type_ == LDataType::FLOAT32) {
    PadWithConstant<float>(src, dst, top, bottom, left, right, pad_type, fill_b_or_gray, fill_g, fill_r);
  } else if (pad_type == PADD_BORDER_CONSTANT && src.data_type_ == LDataType::UINT8) {
    PadWithConstant<uint8_t>(src, dst, top, bottom, left, right, pad_type, fill_b_or_gray, fill_g, fill_r);
  } else if (src.data_type_ == LDataType::FLOAT32) {
    PadImplement<float>(src, dst, top, bottom, left, right, pad_type);
  } else if (src.data_type_ == LDataType::UINT8) {
    PadImplement<uint8_t>(src, dst, top, bottom, left, right, pad_type);
  } else {
    return false;
  }
  return true;
}

std::vector<std::vector<float>> GetDefaultBoxes(const BoxesConfig config) {
  size_t size = config.num_default.size();
  if (size <= 1 || config.feature_size.size() != size || config.steps.size() != size ||
      config.aspect_rations.size() != size) {
    return {};
  }
  if (config.max_scale < config.min_scale) {
    return {};
  }
  std::vector<float> fk;
  float num = static_cast<float>(config.img_shape[0]);
  for (int i = 0; i < config.steps.size(); i++) {
    if (config.steps[i] == 0) {
      return {};
    }
    fk.push_back(num / config.steps[i]);
  }
  float scale_rate = (config.max_scale - config.min_scale) / (config.num_default.size() - 1);
  std::vector<float> scales(config.num_default.size());
  for (int i = 0; i < scales.size(); i++) {
    scales[i] = config.min_scale + scale_rate * i;
  }
  scales.push_back(1.0f);
  std::vector<std::vector<float>> default_boxes;
  for (auto i = 0; i < config.feature_size.size(); i++) {
    float sk1 = scales[i];
    float sk2 = scales[i + 1];
    float sk3 = sqrt(sk1 * sk2);
    std::vector<std::vector<float>> all_sizes;
    float w, h;
    if (i == 0) {
      w = sk1 * sqrt(2);
      h = sk1 / sqrt(2);
      all_sizes = {{0.1, 0.1}, {w, h}, {h, w}};
    } else {
      all_sizes = {{sk1, sk1}};
      for (int j = 0; j < config.aspect_rations[i].size(); j++) {
        w = sk1 * sqrt(config.aspect_rations[i][j]);
        h = sk1 / sqrt(config.aspect_rations[i][j]);
        all_sizes.push_back({w, h});
        all_sizes.push_back({h, w});
      }
      all_sizes.push_back({sk3, sk3});
    }

    for (int j = 0; j < config.feature_size[i]; j++) {
      for (int k = 0; k < config.feature_size[i]; k++) {
        for (int m = 0; m < all_sizes.size(); m++) {
          float cx = (k + 0.5) / fk[i];
          float cy = (j + 0.5) / fk[i];
          default_boxes.push_back({cy, cx, all_sizes[m][1], all_sizes[m][0]});
        }
      }
    }
  }
  return default_boxes;
}

void ConvertBoxes(std::vector<std::vector<float>> &boxes, const std::vector<std::vector<float>> &default_boxes,
                  const BoxesConfig config) {
  constexpr int64_t prior_scaling_size = 2;
  if (boxes.size() != default_boxes.size() || config.prior_scaling.size() != prior_scaling_size) {
    boxes = {};
    return;
  }
  for (int i = 0; i < default_boxes.size(); i++) {
    if (boxes[i].size() != 4 || default_boxes[i].size() != 4) {
      boxes = {};
      return;
    }
    boxes[i][0] = boxes[i][0] * config.prior_scaling[0] * default_boxes[i][2] + default_boxes[i][0];
    boxes[i][1] = boxes[i][1] * config.prior_scaling[0] * default_boxes[i][3] + default_boxes[i][1];
    boxes[i][2] = exp(boxes[i][2] * config.prior_scaling[1]) * default_boxes[i][2];
    boxes[i][3] = exp(boxes[i][3] * config.prior_scaling[1]) * default_boxes[i][3];
  }
}

std::vector<int> ApplyNms(const std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres,
                          int max_boxes) {
  if (all_boxes.size() != all_scores.size()) {
    return {};
  }
  size_t boxes_num = all_boxes.size();
  std::vector<float> areas(boxes_num);
  std::vector<int> order(boxes_num);
  for (auto i = 0; i < boxes_num; i++) {
    if (all_boxes[i].size() < 4) {
      return {};
    }
    areas[i] = (all_boxes[i][3] - all_boxes[i][1] + 1) * (all_boxes[i][2] - all_boxes[i][0] + 1);
    order[i] = i;
  }

  std::sort(order.begin(), order.end(),
            [&all_scores](int pos1, int pos2) { return (all_scores[pos1] > all_scores[pos2]); });
  std::vector<int> keep;
  while (order.size() > 0) {
    int i = order[0];
    keep.push_back(i);
    if (keep.size() >= max_boxes) {
      break;
    }
    int len = order.size() - 1;
    std::vector<float> ovr(len);
    for (int j = 0; j < len; j++) {
      float xx1 = std::max(all_boxes[i][1], all_boxes[order[j + 1]][1]);
      float yy1 = std::max(all_boxes[i][0], all_boxes[order[j + 1]][0]);
      float xx2 = std::min(all_boxes[i][3], all_boxes[order[j + 1]][3]);
      float yy2 = std::min(all_boxes[i][2], all_boxes[order[j + 1]][2]);

      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      ovr[j] = inter / (areas[i] + areas[order[j + 1]] - inter);
    }
    std::vector<int> inds;
    for (int j = 0; j < len; j++) {
      if (ovr[j] <= thres) {
        inds.push_back(j + 1);
      }
    }
    std::vector<int> new_order;
    new_order.reserve(inds.size());
    for (int k = 0; k < inds.size(); k++) {
      new_order.push_back(order[inds[k]]);
    }
    order = new_order;
  }
  return keep;
}

template <typename Pixel_Type>
bool ImplementAffine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> &dsize,
                     Pixel_Type borderValue) {
  if (dsize.size() != 2 || CheckZero(dsize)) {
    return false;
  }

  double IM[6];
  for (int i = 0; i < 6; i++) {
    IM[i] = M[i];
  }

  double D = IM[0] * IM[4] - IM[1] * IM[3];
  D = std::fabs(D) > std::numeric_limits<double>::epsilon() ? 1.0f / D : 0;
  double A11 = IM[4] * D, A22 = IM[0] * D;
  IM[0] = A11;
  IM[1] *= -D;
  IM[3] *= -D;
  IM[4] = A22;
  double b1 = -IM[0] * IM[2] - IM[1] * IM[5];
  double b2 = -IM[3] * IM[2] - IM[4] * IM[5];
  IM[2] = b1;
  IM[5] = b2;
  if (out_img.IsEmpty()) {
    out_img.Init(dsize[0], dsize[1], sizeof(Pixel_Type), src.data_type_);
    RETURN_FALSE_IF_LITEMAT_EMPTY(out_img);
  } else if (out_img.height_ != dsize[1] || out_img.width_ != dsize[0] || out_img.channel_ != src.channel_) {
    return false;
  } else if (out_img.data_type_ != src.data_type_) {
    return false;
  }

  for (int y = 0; y < out_img.height_; y++) {
    for (int x = 0; x < out_img.width_; x++) {
      int src_x = IM[0] * x + IM[1] * y + IM[2];
      int src_y = IM[3] * x + IM[4] * y + IM[5];
      if (src_x >= 0 && src_y >= 0 && src_x < src.width_ && src_y < src.height_) {
        Pixel_Type src_pixel = static_cast<Pixel_Type *>(src.data_ptr_)[src_y * src.width_ + src_x];
        static_cast<Pixel_Type *>(out_img.data_ptr_)[y * out_img.width_ + x] = src_pixel;
      } else {
        static_cast<Pixel_Type *>(out_img.data_ptr_)[y * out_img.width_ + x] = borderValue;
      }
    }
  }

  return true;
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue) {
  if (src.channel_ == 1 && src.data_type_ == LDataType::UINT8) {
    return ImplementAffine(src, out_img, M, dsize, borderValue);
  } else {
    return false;
  }
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue) {
  if (src.channel_ == 3 && src.data_type_ == LDataType::UINT8) {
    return ImplementAffine(src, out_img, M, dsize, borderValue);
  } else {
    return false;
  }
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, FLOAT32_C1 borderValue) {
  if (src.channel_ == 1 && src.data_type_ == LDataType::FLOAT32) {
    return ImplementAffine(src, out_img, M, dsize, borderValue);
  } else {
    return false;
  }
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, FLOAT32_C3 borderValue) {
  constexpr int64_t channel = 3;
  if (src.channel_ == channel && src.data_type_ == LDataType::FLOAT32) {
    return ImplementAffine(src, out_img, M, dsize, borderValue);
  } else {
    return false;
  }
}

inline void RotationMatrix2DImpl(float x, float y, double angle, double scale, LiteMat &M) {
  angle *= CV_PI / 180;
  double alpha = std::cos(angle) * scale;
  double beta = std::sin(angle) * scale;

  M.ptr<double>(0)[0] = alpha;
  M.ptr<double>(0)[1] = beta;
  M.ptr<double>(0)[2] = (1 - alpha) * x - beta * y;
  M.ptr<double>(1)[0] = -beta;
  M.ptr<double>(1)[1] = alpha;
  M.ptr<double>(1)[2] = beta * x + (1 - alpha) * y;
}

bool GetRotationMatrix2D(float x, float y, double angle, double scale, LiteMat &M) {
  M.Init(3, 2, LDataType(LDataType::DOUBLE));
  RETURN_FALSE_IF_LITEMAT_EMPTY(M);
  RotationMatrix2DImpl(x, y, angle, scale, M);
  return true;
}

template <typename T>
bool TransposeImpl(const LiteMat &src, LiteMat &dst) {
  int m = src.width_;
  int n = src.height_;

  dst.Init(n, m, src.data_type_);
  RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst.ptr<T>(i)[j] = src.ptr<T>(j)[i];
    }
  }

  return true;
}

bool Transpose(const LiteMat &src, LiteMat &dst) {
  if (src.IsEmpty()) {
    return false;
  }
  if (src.data_type_ == LDataType::DOUBLE) {
    return TransposeImpl<double>(src, dst);
  } else if (src.data_type_ == LDataType::FLOAT32) {
    return TransposeImpl<float>(src, dst);
  } else {
    return false;
  }
}

template <typename T>
static inline T Hypot_(T a, T b) {
  a = std::abs(a);
  b = std::abs(b);
  if (a > b) {
    b /= a;
    return a * std::sqrt(1 + b * b);
  }

  if (b > 0) {
    a /= b;
    return b * std::sqrt(1 + a * a);
  }
  return 0;
}

template <typename T>
void Calculation(int n, int m, std::vector<double> &W, LiteMat &A, LiteMat &V, const T eps) {
  int max_iter = std::max(m, 30);
  for (int iter = 0; iter < max_iter; iter++) {
    bool change = false;
    T c;
    T s;

    for (int i = 0; i < n - 1; i++) {
      for (int j = i + 1; j < n; j++) {
        T *Ai = A.ptr<T>(i);
        T *Aj = A.ptr<T>(j);
        double a = W[i];
        double p = 0;
        double b = W[j];

        for (int k = 0; k < m; k++) {
          p += static_cast<double>(Ai[k] * Aj[k]);
        }

        if (std::abs(p) <= eps * std::sqrt(static_cast<double>(a * b))) {
          continue;
        }

        p *= 2;
        double beta = a - b;
        double gamma = Hypot_(static_cast<double>(p), beta);

        if (beta < 0) {
          double delta = (gamma - beta) * 0.5;
          s = (T)std::sqrt(delta / gamma);
          c = (T)(p / (gamma * s * 2));
        } else {
          c = (T)std::sqrt((gamma + beta) / (gamma * 2));
          s = (T)(p / (gamma * c * 2));
        }

        a = 0;
        b = 0;
        for (int k = 0; k < m; k++) {
          T t0 = c * Ai[k] + s * Aj[k];
          T t1 = -s * Ai[k] + c * Aj[k];
          Ai[k] = t0;
          Aj[k] = t1;
          a += static_cast<double>(t0 * t0);
          b += static_cast<double>(t1 * t1);
        }
        W[i] = a;
        W[j] = b;
        change = true;
        T *Vi = V.ptr<T>(i);
        T *Vj = V.ptr<T>(j);

        for (int k = 0; k < n; k++) {
          T t0 = c * Vi[k] + s * Vj[k];
          T t1 = -s * Vi[k] + c * Vj[k];
          Vi[k] = t0;
          Vj[k] = t1;
        }
      }
    }

    if (!change) {
      break;
    }
  }
}

template <typename T>
void CalculationMatrix(int n, int m, std::vector<double> &W, LiteMat &A, LiteMat &V, const T eps) {
  for (int i = 0; i < n; i++) {
    double sd = 0.;
    for (int j = 0; j < m; j++) {
      T t = A.ptr<T>(i)[j];
      sd += static_cast<double>(t * t);
    }
    W[i] = sd;

    for (int k = 0; k < n; k++) {
      V.ptr<T>(i)[k] = 0;
    }
    V.ptr<T>(i)[i] = 1;
  }

  Calculation<T>(n, m, W, A, V, eps);
  for (int i = 0; i < n; i++) {
    double sd = 0;
    for (int k = 0; k < m; k++) {
      T t = A.ptr<T>(i)[k];
      sd += static_cast<double>(t * t);
    }
    W[i] = std::sqrt(sd);
  }

  for (int i = 0; i < n - 1; i++) {
    int mid = i;
    for (int j = i + 1; j < n; j++) {
      if (W[mid] < W[j]) {
        mid = j;
      }
    }

    if (i != mid) {
      std::swap(W[i], W[mid]);
      for (int j = 0; j < m; j++) {
        std::swap(A.ptr<T>(i)[j], A.ptr<T>(mid)[j]);
      }

      for (int j = 0; j < n; j++) {
        std::swap(V.ptr<T>(i)[j], V.ptr<T>(mid)[j]);
      }
    }
  }
}

template <typename T>
void JacobiSVD(LiteMat &A, LiteMat &_W, LiteMat &V) {
  double min_val = FLT_MIN;
  T eps = (T)(FLT_EPSILON * 2);
  int m = A.width_;
  int n = _W.height_;
  int urows = m;
  std::vector<double> W(n, 0.);

  CalculationMatrix<T>(n, m, W, A, V, eps);
  for (int i = 0; i < n; i++) {
    _W.ptr<T>(i)[0] = (T)W[i];
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dis(0, 4294967294);

  for (int i = 0; i < urows; i++) {
    double mid = i < n ? W[i] : 0;
    for (int ii = 0; ii < 100 && mid <= min_val; ii++) {
      const T val0 = (T)(1. / m);
      for (int k = 0; k < m; k++) {
        unsigned int rng = dis(gen);
        T val = (rng & 256) != 0 ? val0 : -val0;
        A.ptr<T>(i)[k] = val;
      }

      for (int inner = 0; inner < 2; inner++) {
        for (int j = 0; j < i; j++) {
          mid = 0;
          for (int k = 0; k < m; k++) {
            mid += A.ptr<T>(i)[k] * A.ptr<T>(j)[k];
          }
          T asum = 0;
          for (int k = 0; k < m; k++) {
            T t = (T)(A.ptr<T>(i)[k] - mid * A.ptr<T>(j)[k]);
            A.ptr<T>(i)[k] = t;
            asum += std::abs(t);
          }

          asum = asum > eps * 100 ? 1 / asum : 0;
          for (int k = 0; k < m; k++) {
            A.ptr<T>(i)[k] *= asum;
          }
        }
      }

      mid = 0;
      for (int k = 0; k < m; k++) {
        T t = A.ptr<T>(i)[k];
        mid += static_cast<double>(t * t);
      }
      mid = std::sqrt(mid);
    }

    T s = (T)(mid > min_val ? 1 / mid : 0.);
    for (int k = 0; k < m; k++) {
      A.ptr<T>(i)[k] *= s;
    }
  }
}

template <typename T>
void SVBkSb(int m, int n, LiteMat w, LiteMat u, LiteMat v, const LiteMat src2, LiteMat dst) {
  T eps = DBL_EPSILON * 2;
  double thresgold = 0;
  int nm = std::min(m, n);

  for (int i = 0; i < n; i++) {
    dst.ptr<T>(i)[0] = 0;
  }

  for (int i = 0; i < nm; i++) {
    for (int j = 0; j < w.width_; j++) {
      thresgold += w.ptr<T>(i)[j];
    }
  }
  thresgold *= eps;

  for (int i = 0; i < nm; i++) {
    double wi = w.ptr<T>(i)[0];
    if (static_cast<double>(std::abs(wi)) < thresgold) {
      continue;
    }
    wi = 1 / wi;
    double s = 0;
    for (int j = 0; j < n; j++) {
      s += u.ptr<T>(i)[j] * src2.ptr<T>(j)[0];
    }

    s *= wi;
    for (int j = 0; j < n; j++) {
      dst.ptr<T>(j)[0] = dst.ptr<T>(j)[0] + s * v.ptr<T>(i)[j];
    }
  }
}

bool GetPerspectiveTransformImpl(const LiteMat &src1, const LiteMat &src2, LiteMat dst) {
  LDataType type = src1.data_type_;
  int m = src1.height_;
  int m_ = m;
  int n = src1.width_;

  if (m < n) {
    return false;
  }

  double val_a[64] = {0};
  double val_v[64] = {0};
  double val_w[8] = {0};
  LiteMat a(m_, n, val_a, type);
  Transpose(src1, a);
  LiteMat w(1, n, val_w, type);
  LiteMat v(n, n, val_v, type);
  LiteMat u;

  JacobiSVD<double>(a, w, v);
  u = a;

  SVBkSb<double>(m_, n, w, u, v, src2, dst);
  return true;
}

bool GetPerspectiveTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M) {
  if (src_point.size() != 4 || dst_point.size() != 4) {
    return false;
  }
  double m[8][8];
  double n[8];
  LiteMat src1(8, 8, m, LDataType(LDataType::DOUBLE));
  LiteMat src2(1, 8, n, LDataType(LDataType::DOUBLE));

  for (int i = 0; i < 4; ++i) {
    m[i][0] = m[i + 4][3] = src_point[i].x;
    m[i][1] = m[i + 4][4] = src_point[i].y;
    m[i][2] = m[i + 4][5] = 1;
    m[i][3] = m[i][4] = m[i][5] = m[i + 4][0] = m[i + 4][1] = m[i + 4][2] = 0;
    m[i][6] = -src_point[i].x * dst_point[i].x;
    m[i][7] = -src_point[i].y * dst_point[i].x;
    m[i + 4][6] = -src_point[i].x * dst_point[i].y;
    m[i + 4][7] = -src_point[i].y * dst_point[i].y;
    n[i] = dst_point[i].x;
    n[i + 4] = dst_point[i].y;
  }

  M.Init(3, 3, LDataType(LDataType::DOUBLE));
  RETURN_FALSE_IF_LITEMAT_EMPTY(M);
  LiteMat dst(1, 8, M.data_ptr_, LDataType(LDataType::DOUBLE));

  GetPerspectiveTransformImpl(src1, src2, dst);
  M.ptr<double>(2)[2] = 1;
  return true;
}

bool GetAffineTransformImpl(LiteMat &src, LiteMat &dst) {
  int m = src.height_;
  int n = dst.width_;
  for (int i = 0; i < m; i++) {
    int k = i;
    for (int j = i + 1; j < m; j++) {
      if (std::abs(src.ptr<double>(j)[i]) > std::abs(src.ptr<double>(k)[i])) {
        k = j;
      }
    }

    if (std::abs(src.ptr<double>(k)[i]) < DBL_EPSILON * 100) {
      dst.Init(1, 6, LDataType(LDataType::DOUBLE));
      RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
      (void)memset(dst.data_ptr_, 0, 6 * sizeof(double));
      RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
      return false;
    }
    if (k != i) {
      for (int j = i; j < m; j++) {
        std::swap(src.ptr<double>(i)[j], src.ptr<double>(k)[j]);
      }

      if (dst.data_ptr_) {
        for (int j = 0; j < n; j++) {
          std::swap(dst.ptr<double>(i)[j], dst.ptr<double>(k)[j]);
        }
      }
    }

    const double d = -1 / src.ptr<double>(i)[i];
    for (int j = i + 1; j < m; j++) {
      double alpha = src.ptr<double>(j)[i] * d;
      for (k = i + 1; k < m; k++) {
        src.ptr<double>(j)[k] += alpha * src.ptr<double>(i)[k];
      }

      if (dst.data_ptr_) {
        for (k = 0; k < n; k++) {
          dst.ptr<double>(j)[k] += alpha * dst.ptr<double>(i)[k];
        }
      }
    }
  }

  if (dst.data_ptr_) {
    for (int i = m - 1; i >= 0; i--) {
      for (int j = 0; j < n; j++) {
        double s = dst.ptr<double>(i)[j];
        for (int k = i + 1; k < m; k++) {
          s -= src.ptr<double>(i)[k] * dst.ptr<double>(k)[j];
        }
        dst.ptr<double>(i)[j] = s / src.ptr<double>(i)[i];
      }
    }
  }

  return true;
}

bool GetAffineTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M) {
  if (src_point.size() != 3 || dst_point.size() != 3) {
    return false;
  }
  double m[6 * 6];
  double n[6];
  LiteMat src1(6, 6, m, LDataType(LDataType::DOUBLE));
  LiteMat src2(1, 6, n, LDataType(LDataType::DOUBLE));

  for (int i = 0; i < 3; i++) {
    int j = i * 12;
    int k = i * 12 + 6;
    m[j] = m[k + 3] = src_point[i].x;
    m[j + 1] = m[k + 4] = src_point[i].y;
    m[j + 2] = m[k + 5] = 1;
    m[j + 3] = m[j + 4] = m[j + 5] = 0;
    m[k] = m[k + 1] = m[k + 2] = 0;
    n[i * 2] = dst_point[i].x;
    n[i * 2 + 1] = dst_point[i].y;
  }

  GetAffineTransformImpl(src1, src2);
  M.Init(3, 2, 1, LDataType(LDataType::DOUBLE));
  RETURN_FALSE_IF_LITEMAT_EMPTY(M);
  for (int i = 0; i < M.height_; i++) {
    for (int j = 0; j < M.width_; j++) {
      M.ptr<double>(i)[j] = src2.ptr<double>(i * M.width_ + j)[0];
    }
  }
  return true;
}

bool ConvertRgbToBgr(const LiteMat &src, const LDataType &data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    if (src.IsEmpty()) {
      return false;
    }
    if (mat.IsEmpty()) {
      mat.Init(w, h, 3, LDataType::UINT8);
      RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    }
    if (mat.channel_ != 3) {
      return false;
    }
    if ((src.width_ != w) || (src.height_ != h)) {
      return false;
    }
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = src;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[0] = data_ptr[2];
        ptr[1] = data_ptr[1];
        ptr[2] = data_ptr[0];

        ptr += 3;
        data_ptr += 3;
      }
    }
  } else {
    return false;
  }
  return true;
}

bool ConvertRgbToGray(const LiteMat &src, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    if (src.IsEmpty()) {
      return false;
    }
    if (mat.IsEmpty()) {
      mat.Init(w, h, 1, LDataType::UINT8);
      RETURN_FALSE_IF_LITEMAT_EMPTY(mat);
    }
    if (mat.channel_ != 1) {
      return false;
    }
    if ((src.width_ != w) || (src.height_ != h)) {
      return false;
    }
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = src;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        *ptr = (data_ptr[2] * kB2Gray + data_ptr[1] * kG2Gray + data_ptr[0] * kR2Gray + kGrayShiftDelta) >> kGrayShift;
        ptr++;
        data_ptr += 3;
      }
    }
  } else {
    return false;
  }
  return true;
}

void UpdateOrientationAfineMat(const LiteMat &src, int *rotationDstWidth, int *rotationDstHeight, float (*varM)[2][3],
                               int img_orientation) {
  int srcOrientation = img_orientation;
  if (IM_TOOL_EXIF_ORIENTATION_0_DEG_MIRROR == srcOrientation) {
    (*varM)[0][0] *= -1;
    (*varM)[0][2] += *rotationDstWidth - 1;
  } else if ((IM_TOOL_EXIF_ORIENTATION_180_DEG == srcOrientation) ||
             (IM_TOOL_EXIF_ORIENTATION_180_DEG_MIRROR == srcOrientation)) {
    // 0, 1, 2 is the matrix index of varM
    (*varM)[0][0] = -1;
    (*varM)[0][1] = 0;
    (*varM)[0][2] = *rotationDstWidth - 1;
    (*varM)[1][0] = 0;
    (*varM)[1][1] = -1;
    (*varM)[1][2] = *rotationDstHeight - 1;
    if (IM_TOOL_EXIF_ORIENTATION_180_DEG_MIRROR == srcOrientation) {
      /* with (*varM)irror */
      (*varM)[0][0] *= -1;
      (*varM)[0][2] -= *rotationDstWidth - 1;
    }
  } else if ((IM_TOOL_EXIF_ORIENTATION_90_DEG_MIRROR == srcOrientation) ||
             (IM_TOOL_EXIF_ORIENTATION_90_DEG == srcOrientation)) {
    /* 90 Deg rotation */
    *rotationDstWidth = src.height_;
    *rotationDstHeight = src.width_;
    (*varM)[0][0] = 0;
    (*varM)[0][1] = -1;
    (*varM)[0][2] = *rotationDstWidth - 1;
    (*varM)[1][0] = 1;
    (*varM)[1][1] = 0;
    (*varM)[1][2] = 0;
    if (IM_TOOL_EXIF_ORIENTATION_90_DEG_MIRROR == srcOrientation) {
      /* with Mirror */
      (*varM)[0][1] *= -1;
      (*varM)[0][2] -= *rotationDstWidth - 1;
    }
  } else if ((IM_TOOL_EXIF_ORIENTATION_270_DEG_MIRROR == srcOrientation) ||
             (IM_TOOL_EXIF_ORIENTATION_270_DEG == srcOrientation)) {
    /* 270 Deg rotation */
    *rotationDstWidth = src.height_;
    *rotationDstHeight = src.width_;
    (*varM)[0][0] = 0;
    (*varM)[0][1] = 1;
    (*varM)[0][2] = 0;
    (*varM)[1][0] = -1;
    (*varM)[1][1] = 0;
    (*varM)[1][2] = *rotationDstHeight - 1;
    if (IM_TOOL_EXIF_ORIENTATION_270_DEG_MIRROR == srcOrientation) {
      /* with Mirror */
      (*varM)[0][1] *= -1;
      (*varM)[0][2] += *rotationDstWidth - 1;
    }
  }
}

void ImageToolsConvertImage(const LiteMat &src, const LiteMat &dst, imageToolsImage_t *imageIn,
                            imageToolsImage_t *imageOut) {
  imageIn->image_buff = src.data_ptr_;
  imageIn->h = src.height_;
  imageIn->w = src.width_;
  imageIn->stride = src.width_;
  imageIn->dataType = IM_TOOL_DATA_TYPE_UINT8;

  imageOut->image_buff = dst.data_ptr_;
  imageOut->h = dst.height_;
  imageOut->w = dst.width_;
  imageOut->stride = dst.width_;
  imageOut->dataType = IM_TOOL_DATA_TYPE_FLOAT;
}

int InvAffine2x3(float M[2][3], float invM[][3]) {
  float inv_det = M[0][0] * M[1][1] - M[1][0] * M[0][1];
  if (std::fabs(inv_det) <= std::numeric_limits<float>::epsilon()) {
    return IM_TOOL_RETURN_STATUS_FAILED;
  }
  invM[1][1] = M[0][0] / inv_det;
  invM[0][1] = -M[0][1] / inv_det;
  invM[1][0] = -M[1][0] / inv_det;
  invM[0][0] = M[1][1] / inv_det;
  invM[0][2] = (M[0][1] * M[1][2] - M[1][1] * M[0][2]) / inv_det;
  invM[1][2] = -(M[0][0] * M[1][2] - M[1][0] * M[0][2]) / inv_det;
  return IM_TOOL_RETURN_STATUS_SUCCESS;
}

static float *CalDst(float *dst, float v1, float v2, float v3) {
  *dst++ = v1;
  *dst++ = v2;
  *dst++ = v3;
  return dst;
}

static void ImageWarpAffineHWCFloat(imageToolsImage_t image, imageToolsImage_t warped_image, float invM[2][3]) {
  // 3 is r, g, b
  warped_image.stride *= 3;
  image.stride *= 3;

  float *warped_image_buff = reinterpret_cast<float *>(warped_image.image_buff);

  float *image_buff = reinterpret_cast<float *>(image.image_buff);
  for (int y0 = 0; y0 < warped_image.h; y0++) {
    // Init pointers to start of rows
    float *dst = warped_image_buff + y0 * warped_image.stride;

    for (int x0 = 0; x0 < warped_image.w; x0++) {
      // number 0, 1, 2 is the index of MATRIX 'invM'
      float fPosx = (static_cast<float>(x0) * invM[0][0]) + (static_cast<float>(y0) * invM[0][1]) + invM[0][2];
      float fPosy = (static_cast<float>(x0) * invM[1][0]) + (static_cast<float>(y0) * invM[1][1]) + invM[1][2];
      int iPosy = static_cast<int>(fPosy + 2) - 2;  // for floor like result until -2.
      int iPosx = static_cast<int>(fPosx + 2) - 2;  // for floor like result until -2.
      if ((iPosx < -1) || (iPosx >= image.w) || (iPosy < -1) || (iPosy >= image.h)) {
        dst = CalDst(dst, 0.0f, 0.0f, 0.0f);
        continue;
      }
      float fRsiduy = fPosy - iPosy;
      float fRsidux = fPosx - iPosx;
      float fOut0 = 0;
      float fOut1 = 0;
      float fOut2 = 0;
      float *fTopeLeft = image_buff + iPosy * image.stride + iPosx * 3;
      float fCoeff = 1 - fRsidux - fRsiduy + fRsidux * fRsiduy;
      if ((iPosx >= 0) && (iPosy >= 0)) {
        // number 0, 1, 2 is the index of MATRIX 'fTopeLeft'
        fOut0 += fCoeff * fTopeLeft[0];
        fOut1 += fCoeff * fTopeLeft[1];
        fOut2 += fCoeff * fTopeLeft[2];
      }
      float fSum = fCoeff;
      fCoeff = fRsiduy - fRsidux * fRsiduy;
      if ((iPosx >= 0) && (iPosy < image.h - 1)) {
        // Image channel G and B could be accessed by adding number of 1, 2
        fOut0 += fCoeff * fTopeLeft[image.stride];
        fOut1 += fCoeff * fTopeLeft[image.stride + 1];
        fOut2 += fCoeff * fTopeLeft[image.stride + 2];
      }
      fSum += fCoeff;
      fCoeff = fRsidux - fRsidux * fRsiduy;
      if ((iPosx < image.w - 1) && (iPosy >= 0)) {
        // Image channel G and B could be accessed by adding number of 1, 2
        fOut0 += fCoeff * fTopeLeft[3];
        fOut1 += fCoeff * fTopeLeft[3 + 1];
        fOut2 += fCoeff * fTopeLeft[3 + 2];
      }
      fSum += fCoeff;
      if ((iPosx < image.w - 1) && (iPosy < image.h - 1)) {
        // Image channel G and B could be accessed by adding number of 1, 2
        fOut0 += (1 - fSum) * fTopeLeft[image.stride + 3];
        fOut1 += (1 - fSum) * fTopeLeft[image.stride + 3 + 1];
        fOut2 += (1 - fSum) * fTopeLeft[image.stride + 3 + 2];
      }
      dst = CalDst(dst, fOut0, fOut1, fOut2);
    }
  }
}

static void ImageWarpAffineHWCUint8(imageToolsImage_t image, imageToolsImage_t warped_image, float invM[2][3]) {
  // 3 is r, g, b
  warped_image.stride *= 3;
  image.stride *= 3;
  float *warped_image_buff = reinterpret_cast<float *>(warped_image.image_buff);

  uint8_t *image_buff = reinterpret_cast<uint8_t *>(image.image_buff);
  for (int y0 = 0; y0 < warped_image.h; y0++) {
    // Init pointers to start of rows
    float *dst = warped_image_buff + y0 * warped_image.stride;

    for (int x0 = 0; x0 < warped_image.w; x0++) {
      float fPosx = (static_cast<float>(x0) * invM[0][0]) + (static_cast<float>(y0) * invM[0][1]) + invM[0][2];
      float fPosy = (static_cast<float>(x0) * invM[1][0]) + (static_cast<float>(y0) * invM[1][1]) + invM[1][2];

      int iPosy = static_cast<int>(fPosy + 2) - 2;  // for floor like result until -2.
      int iPosx = static_cast<int>(fPosx + 2) - 2;  // for floor like result until -2.
      if ((iPosx < -1) || (iPosx >= image.w) || (iPosy < -1) || (iPosy >= image.h)) {
        dst = CalDst(dst, 0.0f, 0.0f, 0.0f);
        continue;
      }
      float fRsiduy = fPosy - iPosy;
      float fRsidux = fPosx - iPosx;
      float fOut0 = 0;
      float fOut1 = 0;
      float fOut2 = 0;
      uint8_t *uiTopeLeft = image_buff + iPosy * image.stride + iPosx * 3;
      float fCoeff = 1 - fRsidux - fRsiduy + fRsidux * fRsiduy;
      if ((iPosx >= 0) && (iPosy >= 0)) {
        // number 0, 1, 2 is the index of MATRIX round.
        fOut0 += fCoeff * static_cast<float>(uiTopeLeft[0]);
        fOut1 += fCoeff * static_cast<float>(uiTopeLeft[1]);
        fOut2 += fCoeff * static_cast<float>(uiTopeLeft[2]);
      }
      float fSum = fCoeff;
      fCoeff = fRsiduy - fRsidux * fRsiduy;
      if ((iPosx >= 0) && (iPosy < image.h - 1)) {
        fOut0 += fCoeff * static_cast<float>(uiTopeLeft[image.stride]);
        fOut1 += fCoeff * static_cast<float>(uiTopeLeft[image.stride + 1]);
        fOut2 += fCoeff * static_cast<float>(uiTopeLeft[image.stride + 2]);
      }
      fSum += fCoeff;
      fCoeff = fRsidux - fRsidux * fRsiduy;
      if ((iPosx < image.w - 1) && (iPosy >= 0)) {
        fOut0 += fCoeff * static_cast<float>(uiTopeLeft[3]);
        fOut1 += fCoeff * static_cast<float>(uiTopeLeft[3 + 1]);
        fOut2 += fCoeff * static_cast<float>(uiTopeLeft[3 + 2]);
      }
      fSum += fCoeff;
      if ((iPosx < image.w - 1) && (iPosy < image.h - 1)) {
        fOut0 += (1 - fSum) * static_cast<float>(uiTopeLeft[image.stride + 3]);
        fOut1 += (1 - fSum) * static_cast<float>(uiTopeLeft[image.stride + 3 + 1]);
        fOut2 += (1 - fSum) * static_cast<float>(uiTopeLeft[image.stride + 3 + 2]);
      }
      dst = CalDst(dst, fOut0, fOut1, fOut2);
    }
  }
}

int ImageWarpAffineHWC(imageToolsImage_t image, imageToolsImage_t warped_image, float M[2][3], bool bIsMInv) {
  if ((IM_TOOL_DATA_TYPE_FLOAT != warped_image.dataType) ||
      ((IM_TOOL_DATA_TYPE_FLOAT != image.dataType) && (IM_TOOL_DATA_TYPE_UINT8 != image.dataType))) {
    return IM_TOOL_RETURN_STATUS_INVALID_INPUT;
  }
  float invM[2][3];
  if (bIsMInv) {
    for (int iy = 0; iy < 2; iy++) {
      for (int ix = 0; ix < 3; ix++) {
        invM[iy][ix] = M[iy][ix];
      }
    }
  } else {
    if (InvAffine2x3(M, invM) != IM_TOOL_RETURN_STATUS_SUCCESS) {
      return IM_TOOL_RETURN_STATUS_FAILED;
    }
  }

  if (IM_TOOL_DATA_TYPE_FLOAT == image.dataType) {
    ImageWarpAffineHWCFloat(image, warped_image, invM);
  } else {
    ImageWarpAffineHWCUint8(image, warped_image, invM);
  }
  return IM_TOOL_RETURN_STATUS_SUCCESS;
}

bool ResizePreserveARWithFiller(LiteMat &src, LiteMat &dst, int h, int w, float (*ratioShiftWShiftH)[3],
                                float (*invM)[2][3], int img_orientation) {
  if (src.IsEmpty() || src.channel_ != 3 || h <= 0 || w <= 0 || h > 10000 || w > 10000) {
    return false;
  }
  if (ratioShiftWShiftH == nullptr || invM == nullptr) {
    return false;
  }
  if (dst.IsEmpty()) {
    dst.Init(w, h, src.channel_, LDataType::FLOAT32);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  }

  float varM[2][3] = {{1.0, 0, 0}, {0, 1.0, 0}};
  const float divisor = 2.0;
  int rotationDstWidth = src.width_;
  int rotationDstHeight = src.height_;
  if (rotationDstWidth == 0 || rotationDstHeight == 0) {
    return false;
  }

  if (dst.height_ == 0) {
    return false;
  }

  if (img_orientation > IM_TOOL_EXIF_ORIENTATION_0_DEG) {
    UpdateOrientationAfineMat(src, &rotationDstWidth, &rotationDstHeight, &varM, img_orientation);
  }

  /* Resize after orientation fix */
  float srcAR = static_cast<float>(rotationDstWidth) / static_cast<float>(rotationDstHeight);
  float dstAR = static_cast<float>(dst.width_) / static_cast<float>(dst.height_);
  auto dstActiveWidth = static_cast<float>(dst.width_);
  auto dstActiveHeight = static_cast<float>(dst.height_);
  float ratio, shiftW, shiftH;
  if (srcAR < dstAR) {
    ratio = static_cast<float>(dst.height_) / static_cast<float>(rotationDstHeight);
    dstActiveWidth = static_cast<float>(rotationDstWidth) * ratio;
  } else {
    ratio = static_cast<float>(dst.width_) / static_cast<float>(rotationDstWidth);
    dstActiveHeight = static_cast<float>(rotationDstHeight) * ratio;
  }
  shiftW = (static_cast<float>(dst.width_) - dstActiveWidth) / divisor;
  shiftH = (static_cast<float>(dst.height_) - dstActiveHeight) / divisor;
  for (auto &iy : varM) {
    for (float &ix : iy) {
      // cppcheck-suppress useStlAlgorithm
      ix *= ratio;
    }
  }

  varM[0][2] += shiftW;
  varM[1][2] += shiftH;
  /* Resize and shift by affine transform  */
  imageToolsImage_t imageIn, imageOut;
  ImageToolsConvertImage(src, dst, &imageIn, &imageOut);
  if (InvAffine2x3(varM, *invM) != IM_TOOL_RETURN_STATUS_SUCCESS) {
    return false;
  }
  int retVal = ImageWarpAffineHWC(imageIn, imageOut, *invM, true);
  if (retVal != 0) {
    return false;
  }

  // 0, 1, 2 is the index of corresponding elem in ratioShiftWShiftH
  (*ratioShiftWShiftH)[0] = ratio;
  (*ratioShiftWShiftH)[1] = shiftW;
  (*ratioShiftWShiftH)[2] = shiftH;

  return true;
}

template <typename T>
void HWC2CHWImpl(const T *src_ptr, T *dst_ptr, int height, int width, int channel) {
  int stride = width * height;
  for (int i = 0; i != stride; ++i) {
    for (int c = 0; c != channel; ++c) {
      dst_ptr[c * stride + i] = src_ptr[i * channel + c];
    }
  }
}

bool HWC2CHW(LiteMat &src, LiteMat &dst) {
  if (src.IsEmpty()) {
    return false;
  }
  if (dst.IsEmpty() || dst.width_ != src.height_ || dst.height_ != src.channel_ || dst.channel_ != src.width_ ||
      dst.data_type_ != src.data_type_) {
    dst.Init(src.height_, src.channel_, src.width_, src.data_type_);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  }
  if (src.data_type_ == LDataType::FLOAT32) {
    HWC2CHWImpl<float>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::UINT8) {
    HWC2CHWImpl<uint8_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::INT16) {
    HWC2CHWImpl<int16_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::INT32) {
    HWC2CHWImpl<int32_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::INT64) {
    HWC2CHWImpl<int64_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::UINT16) {
    HWC2CHWImpl<uint16_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::UINT32) {
    HWC2CHWImpl<uint32_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::UINT64) {
    HWC2CHWImpl<uint64_t>(src, dst, src.height_, src.width_, src.channel_);
  } else if (src.data_type_ == LDataType::DOUBLE) {
    HWC2CHWImpl<double>(src, dst, src.height_, src.width_, src.channel_);
  } else {
    return false;
  }
  return true;
}

}  // namespace dataset
}  // namespace mindspore
