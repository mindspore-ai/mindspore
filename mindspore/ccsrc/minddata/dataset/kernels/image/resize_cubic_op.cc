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
#include "minddata/dataset/kernels/image/resize_cubic_op.h"

namespace mindspore {
namespace dataset {
// using 8 bits for result
constexpr uint8_t PrecisionBits = 22;

// construct lookup table
static const std::vector<uint8_t> _clip8_table = []() {
  std::vector<uint8_t> v1(896, 0);
  std::vector<uint8_t> v2(384, 255);
  for (int i = 0; i < 256; i++) {
    v1[i + 640] = i;
  }
  v1.insert(v1.end(), v2.begin(), v2.end());
  return v1;
}();

static const uint8_t *clip8_table = &_clip8_table[640];

static inline uint8_t clip8(unsigned int input) { return clip8_table[input >> PrecisionBits]; }

static inline double cubic_interp(double x) {
  double a = -0.5;
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  }
  if (x < 2.0) {
    return (((x - 5) * x + 8) * x - 4) * a;
  }
  return 0.0;
}

struct interpolation {
  double (*interpolation)(double x);
  double threshold;
};

int calc_coeff(int input_size, int out_size, int input0, int input1, struct interpolation *interp,
               std::vector<int> &regions, std::vector<double> &coeffs_interp) {
  double threshold, scale, interp_scale;
  int kernel_size;

  if (out_size == 0) {
    MS_LOG(ERROR) << "out_size can not be zero.";
    return 0;
  }
  scale = static_cast<double>((input1 - input0)) / out_size;
  if (scale < 1.0) {
    interp_scale = 1.0;
  } else {
    interp_scale = scale;
  }

  // obtain size
  threshold = interp->threshold * interp_scale;

  // coefficients number
  kernel_size = static_cast<int>(ceil(threshold)) * 2 + 1;
  if (out_size > INT_MAX / (kernel_size * static_cast<int>(sizeof(double)))) {
    MS_LOG(WARNING) << "Unable to allocator memory as output Image size is so large.";
    return 0;
  }

  // coefficient array
  std::vector<double> coeffs(out_size * kernel_size, 0.0);
  std::vector<int> region(out_size * 2, 0);

  for (int xx = 0; xx < out_size; xx++) {
    double center = input0 + (xx + 0.5) * scale;
    double mm = 0.0, ss = 1.0 / interp_scale;
    int x;
    // Round for x_min
    int x_min = static_cast<int>((center - threshold + 0.5));
    if (x_min < 0) {
      x_min = 0;
    }
    // Round for x_max
    int x_max = static_cast<int>((center + threshold + 0.5));
    if (x_max > input_size) {
      x_max = input_size;
    }
    x_max -= x_min;
    double *coeff = &coeffs[xx * kernel_size];
    for (x = 0; x < x_max; x++) {
      double m = interp->interpolation((x + x_min - center + 0.5) * ss);
      coeff[x] = m;
      mm += m;
    }
    for (x = 0; x < x_max; x++) {
      if (mm != 0.0) {
        coeff[x] /= mm;
      }
    }
    // Remaining values should stay empty if they are used despite of x_max.
    for (; x < kernel_size; x++) {
      coeff[x] = 0;
    }
    region[xx * 2] = x_min;
    region[xx * 2 + 1] = x_max;
  }

  regions = std::move(region);
  coeffs_interp = std::move(coeffs);
  return kernel_size;
}

void normalize_coeff(int out_size, int kernel_size, const std::vector<double> &prekk, std::vector<int> &kk) {
  for (int x = 0; x < out_size * kernel_size; x++) {
    if (prekk[x] < 0) {
      kk[x] = static_cast<int>((-0.5 + prekk[x] * (1 << PrecisionBits)));
    } else {
      kk[x] = static_cast<int>((0.5 + prekk[x] * (1 << PrecisionBits)));
    }
  }
}

Status ImagingHorizontalInterp(LiteMat &output, LiteMat input, int offset, int kernel_size,
                               const std::vector<int> &regions, const std::vector<double> &prekk) {
  int ss0, ss1, ss2;
  int32_t *k = nullptr;

  // normalize previous calculated coefficients
  std::vector<int> kk(prekk.begin(), prekk.end());
  normalize_coeff(output.width_, kernel_size, prekk, kk);
  uint8_t *input_ptr = input;
  uint8_t *output_ptr = output;
  int32_t input_width = input.width_ * 3;
  int32_t output_width = output.width_ * 3;

  for (int yy = 0; yy < output.height_; yy++) {
    // obtain the ptr of output, and put calculated value into it
    uint8_t *bgr_buf = output_ptr;
    for (int xx = 0; xx < output.width_; xx++) {
      int x_min = regions[xx * 2];
      int x_max = regions[xx * 2 + 1];
      k = &kk[xx * kernel_size];
      ss0 = ss1 = ss2 = 1 << (PrecisionBits - 1);
      for (int x = 0; x < x_max; x++) {
        ss0 += (input_ptr[(yy + offset) * input_width + (x + x_min) * 3]) * k[x];
        ss1 += (input_ptr[(yy + offset) * input_width + (x + x_min) * 3 + 1]) * k[x];
        ss2 += (input_ptr[(yy + offset) * input_width + (x + x_min) * 3 + 2]) * k[x];
      }
      bgr_buf[0] = clip8(ss0);
      bgr_buf[1] = clip8(ss1);
      bgr_buf[2] = clip8(ss2);
      bgr_buf += 3;
    }
    output_ptr += output_width;
  }
  return Status::OK();
}

Status ImagingVerticalInterp(LiteMat &output, LiteMat input, int kernel_size, const std::vector<int> &regions,
                             const std::vector<double> &prekk) {
  int ss0, ss1, ss2;

  // normalize previous calculated coefficients
  std::vector<int> kk(prekk.begin(), prekk.end());
  normalize_coeff(output.height_, kernel_size, prekk, kk);
  uint8_t *input_ptr = input;
  uint8_t *output_ptr = output;
  const int32_t input_width = input.width_ * 3;
  const int32_t output_width = output.width_ * 3;

  for (int yy = 0; yy < output.height_; yy++) {
    // obtain the ptr of output, and put calculated value into it
    uint8_t *bgr_buf = output_ptr;
    int32_t *k = &kk[yy * kernel_size];
    int y_min = regions[yy * 2];
    int y_max = regions[yy * 2 + 1];
    for (int xx = 0; xx < output.width_; xx++) {
      ss0 = ss1 = ss2 = 1 << (PrecisionBits - 1);
      for (int y = 0; y < y_max; y++) {
        ss0 += (input_ptr[(y + y_min) * input_width + xx * 3]) * k[y];
        ss1 += (input_ptr[(y + y_min) * input_width + xx * 3 + 1]) * k[y];
        ss2 += (input_ptr[(y + y_min) * input_width + xx * 3 + 2]) * k[y];
      }
      bgr_buf[0] = clip8(ss0);
      bgr_buf[1] = clip8(ss1);
      bgr_buf[2] = clip8(ss2);
      bgr_buf += 3;
    }
    output_ptr += output_width;
  }
  return Status::OK();
}

bool ImageInterpolation(LiteMat input, LiteMat &output, int x_size, int y_size, struct interpolation *interp,
                        const int rect[4]) {
  int horizontal_interp, vertical_interp, horiz_kernel, vert_kernel, rect_y0, rect_y1;
  std::vector<int> horiz_region, vert_region;
  std::vector<double> horiz_coeff, vert_coeff;
  LiteMat temp;

  horizontal_interp = x_size != input.width_ || rect[2] != x_size || rect[0];
  vertical_interp = y_size != input.height_ || rect[3] != y_size || rect[1];

  horiz_kernel = calc_coeff(input.width_, x_size, rect[0], rect[2], interp, horiz_region, horiz_coeff);
  if (!horiz_kernel) {
    return false;
  }

  vert_kernel = calc_coeff(input.height_, y_size, rect[1], rect[3], interp, vert_region, vert_coeff);
  if (!vert_kernel) {
    return false;
  }

  // first and last used row in the input image
  rect_y0 = vert_region[0];
  rect_y1 = vert_region[y_size * 2 - 1] + vert_region[y_size * 2 - 2];

  // two-direction resize, horizontal resize
  if (horizontal_interp) {
    // Shift region for vertical resize
    for (int i = 0; i < y_size; i++) {
      vert_region[i * 2] -= rect_y0;
    }
    temp.Init(x_size, rect_y1 - rect_y0, 3, LDataType::UINT8, false);

    auto rc = ImagingHorizontalInterp(temp, input, rect_y0, horiz_kernel, horiz_region, horiz_coeff);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "Image horizontal resize failed, error msg is " << rc;
      return false;
    }
    if (temp.IsEmpty()) {
      return false;
    }
    output = input = temp;
  }

  /* vertical resize */
  if (vertical_interp) {
    output.Init(input.width_, y_size, 3, LDataType::UINT8, false);
    if (!output.IsEmpty()) {
      auto rc = ImagingVerticalInterp(output, input, vert_kernel, vert_region, vert_coeff);
      if (rc.IsError()) {
        MS_LOG(ERROR) << "Image vertical resize failed, error msg is " << rc;
        return false;
      }
    }
    if (output.IsEmpty()) {
      return false;
    }
  }
  if (!horizontal_interp && !vertical_interp) {
    output = input;
  }
  return true;
}

bool ResizeCubic(const LiteMat &input, LiteMat &dst, int dst_w, int dst_h) {
  if (input.data_type_ != LDataType::UINT8 || input.channel_ != 3) {
    MS_LOG(ERROR) << "Unsupported data type, only support input image of uint8 dtype and 3 channel, got channel: " +
                       std::to_string(input.channel_);
    return false;
  }
  int x_size = dst_w, y_size = dst_h;
  int rect[4] = {0, 0, input.width_, input.height_};
  LiteMat output;

  struct interpolation interp = {cubic_interp, 2.0};
  bool res = ImageInterpolation(input, output, x_size, y_size, &interp, rect);

  auto ret_code = memcpy_s(dst.data_ptr_, output.size_, output.data_ptr_, output.size_);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "memcpy_s failed when copying tensor.";
    return false;
  }
  return res;
}
}  // namespace dataset
}  // namespace mindspore
