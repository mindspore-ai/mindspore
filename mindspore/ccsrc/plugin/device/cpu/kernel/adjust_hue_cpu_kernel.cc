/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/adjust_hue_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdjustHueInputNum = 2;
constexpr size_t kAdjustHueOutputNum = 1;
const std::int64_t kAdjustHueParallelNum = 8 * 1024;
const std::int64_t kAdjustHueZero = 0;
const std::int64_t kAdjustHueOne = 1;
const std::int64_t kAdjustHueTwo = 2;
const std::int64_t kAdjustHueThree = 3;
const std::int64_t kAdjustHueFour = 4;
const std::int64_t kAdjustHueFive = 5;
}  // namespace

namespace detail {
static void rgb_to_hv_range(float r, float g, float b, float *h, float *v_min, float *v_max) {
  float v_mid;
  int h_category;
  const float eps = 1e-6;
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  // For the conditions, we don't care about the case where two components are
  // equal. It is okay to count it in either side in that case.
  if (r < g) {
    if (b < r) {
      // b < r < g
      *v_max = g;
      v_mid = r;
      *v_min = b;
      h_category = kAdjustHueOne;
    } else if (b > g) {
      // r < g < b
      *v_max = b;
      v_mid = g;
      *v_min = r;
      h_category = kAdjustHueThree;
    } else {
      // r < b < g
      *v_max = g;
      v_mid = b;
      *v_min = r;
      h_category = kAdjustHueTwo;
    }
  } else {
    // g < r
    if (b < g) {
      // b < g < r
      *v_max = r;
      v_mid = g;
      *v_min = b;
      h_category = kAdjustHueZero;
    } else if (b > r) {
      // g < r < b
      *v_max = b;
      v_mid = r;
      *v_min = g;
      h_category = kAdjustHueFour;
    } else {
      // g < b < r
      *v_max = r;
      v_mid = b;
      *v_min = g;
      h_category = kAdjustHueFive;
    }
  }
  if (std::fabs(*v_max - *v_min) <= eps) {
    *h = 0;
    return;
  }
  auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
  bool increase = h_category % 2 == 0;
  *h = h_category + (increase ? ratio : (1 - ratio));
}

// Helper function to convert from H-and-V-range to RGB.
template <typename T>
static void hv_range_to_rgb(float h, float v_min, float v_max, T *r, T *g, T *b) {
  int h_category = static_cast<int>(h);
  float ratio = h - h_category;
  bool increase = h_category % 2 == 0;
  if (!increase) {
    ratio = 1 - ratio;
  }
  float v_mid = v_min + ratio * (v_max - v_min);
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  switch (h_category) {
    case kAdjustHueZero:
      *r = static_cast<T>(v_max);
      *g = static_cast<T>(v_mid);
      *b = static_cast<T>(v_min);
      break;
    case kAdjustHueOne:
      *r = static_cast<T>(v_mid);
      *g = static_cast<T>(v_max);
      *b = static_cast<T>(v_min);
      break;
    case kAdjustHueTwo:
      *r = static_cast<T>(v_min);
      *g = static_cast<T>(v_max);
      *b = static_cast<T>(v_mid);
      break;
    case kAdjustHueThree:
      *r = static_cast<T>(v_min);
      *g = static_cast<T>(v_mid);
      *b = static_cast<T>(v_max);
      break;
    case kAdjustHueFour:
      *r = static_cast<T>(v_mid);
      *g = static_cast<T>(v_min);
      *b = static_cast<T>(v_max);
      break;
    case kAdjustHueFive:
    default:
      *r = static_cast<T>(v_max);
      *g = static_cast<T>(v_min);
      *b = static_cast<T>(v_mid);
  }
}

HsvTuple rgb2hsv(const float r, const float g, const float b) {
  HsvTuple tuple;
  const float M = fmaxf(r, fmaxf(g, b));
  const float m = fminf(r, fminf(g, b));
  const float chroma = M - m;
  const float eps = 1e-6;
  float h = 0.0f, s = 0.0f;
  // hue
  if (chroma > 0.0f) {
    if (std::fabs(M - r) <= eps) {
      const float num = (g - b) / chroma;
      const float sign = copysignf(1.0f, num);
      h = (static_cast<float>(sign < 0.0f) * 6.0f + sign * fmodf(sign * num, 6.0f)) / 6.0f;
    } else if (std::fabs(M - g) <= eps) {
      h = ((b - r) / chroma + 2.0f) / 6.0f;
    } else {
      h = ((r - g) / chroma + 4.0f) / 6.0f;
    }
  } else {
    h = 0.0f;
  }
  // saturation
  if (M > 0) {
    s = chroma / M;
  } else {
    s = 0.0f;
  }
  tuple.h = h;
  tuple.s = s;
  tuple.v = M;
  return tuple;
}

RgbTuple hsv2rgb(const float h, const float s, const float v) {
  RgbTuple tuple;
  const float new_h = h * 6.0f;
  const float chroma = v * s;
  const float x = chroma * (1.0f - fabsf(fmodf(new_h, 2.0f) - 1.0f));
  const float new_m = v - chroma;
  const bool between_0_and_1 = new_h >= 0.0f && new_h < 1.0f;
  const bool between_1_and_2 = new_h >= 1.0f && new_h < 2.0f;
  const bool between_2_and_3 = new_h >= 2.0f && new_h < 3.0f;
  const bool between_3_and_4 = new_h >= 3.0f && new_h < 4.0f;
  const bool between_4_and_5 = new_h >= 4.0f && new_h < 5.0f;
  const bool between_5_and_6 = new_h >= 5.0f && new_h < 6.0f;
  tuple.r = chroma * static_cast<float>(between_0_and_1 || between_5_and_6) +
            x * static_cast<float>(between_1_and_2 || between_4_and_5) + new_m;
  tuple.g = chroma * static_cast<float>(between_1_and_2 || between_2_and_3) +
            x * static_cast<float>(between_0_and_1 || between_3_and_4) + new_m;
  tuple.b = chroma * static_cast<float>(between_3_and_4 || between_4_and_5) +
            x * static_cast<float>(between_2_and_3 || between_5_and_6) + new_m;
  return tuple;
}

template <typename T>
bool LaunchAdjustHueKernel(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &outputs) {
  auto input_data = static_cast<T *>(inputs[0]->addr);
  auto output_data = static_cast<T *>(outputs[0]->addr);
  auto delta_h = static_cast<std::float_t *>(inputs[1]->addr)[0];
  std::int64_t num_elements = SizeToLong(inputs[0]->size / sizeof(T));
  constexpr int64_t kChannelSize = 3;
  auto sharder_adjusthue = [input_data, delta_h, output_data, kChannelSize](int64_t start, int64_t end) {
    for (int64_t i = start * kChannelSize; i < end * kChannelSize; i = i + kChannelSize) {
      // CPU compute
      float h, v_min, v_max;
      rgb_to_hv_range(static_cast<float>(*(input_data + i)), static_cast<float>(*(input_data + i + 1)),
                      static_cast<float>(*(input_data + i + 2)), &h, &v_min, &v_max);

      static const int kChannelRange = 6;
      // Adjust the hue value. And adjust the hue back into the valid
      // range of [0, 6). It is faster than a fmod by avoiding
      // a float-point division since h is often very close to this
      // range.
      h += delta_h * kChannelRange;
      while (h < 0) {
        h += kChannelRange;
      }
      while (h >= kChannelRange) {
        h -= kChannelRange;
      }

      hv_range_to_rgb<T>(h, v_min, v_max, &output_data[i], &output_data[i + 1], &output_data[i + 2]);
    }
  };
  std::int64_t total = num_elements / kChannelSize;
  std::int64_t per_unit_size{total / std::min(kAdjustHueParallelNum - SizeToLong(kAdjustHueInputNum), total)};
  if (total > kAdjustHueParallelNum) {
    CPUKernelUtils::ParallelFor(sharder_adjusthue, static_cast<size_t>(total), static_cast<float>(per_unit_size));
  } else {
    sharder_adjusthue(0, total);
  }
  return true;
}

template <typename T>
bool LaunchAdjustHueKernelHalf(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &outputs) {
  auto input_data = static_cast<T *>(inputs[0]->addr);
  auto output_data = static_cast<T *>(outputs[0]->addr);
  auto delta_h = static_cast<std::float_t *>(inputs[1]->addr)[0];
  std::int64_t num_elements = SizeToLong(inputs[0]->size / sizeof(T));
  constexpr int64_t kChannelSize = 3;
  auto sharder_adjusthue = [input_data, delta_h, output_data, kChannelSize](int64_t start, int64_t end) {
    for (int64_t i = start * kChannelSize; i < end * kChannelSize; i = i + kChannelSize) {
      const HsvTuple hsv = rgb2hsv(static_cast<float>(*(input_data + i)), static_cast<float>(*(input_data + i + 1)),
                                   static_cast<float>(*(input_data + i + 2)));
      float new_h = hsv.h;
      float new_s = hsv.s;
      float new_v = hsv.v;
      // hue adjustment
      new_h = fmodf(hsv.h + delta_h, 1.0f);
      if (new_h < 0.0f) {
        new_h = fmodf(1.0f + new_h, 1.0f);
      }
      const RgbTuple rgb = hsv2rgb(new_h, new_s, new_v);
      output_data[i] = static_cast<T>(rgb.r);
      output_data[i + 1] = static_cast<T>(rgb.g);
      output_data[i + 2] = static_cast<T>(rgb.b);
    }
  };
  std::int64_t total = num_elements / kChannelSize;
  std::int64_t per_unit_size{total / std::min(kAdjustHueParallelNum - SizeToLong(kAdjustHueInputNum), total)};
  if (total > kAdjustHueParallelNum) {
    CPUKernelUtils::ParallelFor(sharder_adjusthue, static_cast<size_t>(total), static_cast<float>(per_unit_size));
  } else {
    sharder_adjusthue(0, total);
  }
  return true;
}
}  // namespace detail

bool AdjustHueCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdjustHueInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdjustHueOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

bool AdjustHueCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  switch (dtype_) {
    case kNumberTypeFloat16:
      (void)detail::LaunchAdjustHueKernelHalf<Eigen::half>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      (void)detail::LaunchAdjustHueKernel<float>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "For AdjustHue, the type of 'image' should be float16, float32, but got "
                        << TypeIdLabel(dtype_) << ".";
  }
  return true;
}

std::vector<KernelAttr> AdjustHueCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdjustHue, AdjustHueCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
