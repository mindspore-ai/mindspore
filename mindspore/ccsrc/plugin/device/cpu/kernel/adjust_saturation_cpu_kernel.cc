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

#include "plugin/device/cpu/kernel/adjust_saturation_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const std::int64_t kAdjustSaturationParallelNum = 64 * 1024;
const std::int64_t kAdjustSaturationZero = 0;
const std::int64_t kAdjustSaturationOne = 1;
const std::int64_t kAdjustSaturationTwo = 2;
const std::int64_t kAdjustSaturationThree = 3;
const std::int64_t kAdjustSaturationFour = 4;
const std::int64_t kAdjustSaturationFive = 5;
const std::float_t kAdjustSaturationSix = 6;
}  // namespace

namespace detail {
static void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v) {
  float vv = std::max(r, std::max(g, b));
  float range = vv - std::min(r, std::min(g, b));
  const float eps = 1e-6;
  if (vv > 0) {
    *s = range / vv;
  } else {
    *s = 0;
  }
  float norm = kAdjustSaturationOne / (kAdjustSaturationSix * range);
  float hh;
  if (std::fabs(r - vv) <= eps) {
    hh = norm * (g - b);
  } else if (std::fabs(g - vv) <= eps) {
    hh = norm * (b - r) + kAdjustSaturationTwo / kAdjustSaturationSix;
  } else {
    hh = norm * (r - g) + kAdjustSaturationFour / kAdjustSaturationSix;
  }
  if (range <= 0.0) {
    hh = 0;
  }
  if (hh < 0.0) {
    hh = hh + kAdjustSaturationOne;
  }
  *v = vv;
  *h = hh;
}

template <typename T>
static void hsv_to_rgb(float h, float s, float v, T *r, T *g, T *b) {
  float c = s * v;
  float m = v - c;
  float dh = h * kAdjustSaturationSix;
  float rr, gg, bb;
  int h_category = static_cast<int>(dh);
  float fmodu = dh;
  while (fmodu <= 0) {
    fmodu += kAdjustSaturationTwo;
  }
  while (fmodu >= kAdjustSaturationTwo) {
    fmodu -= kAdjustSaturationTwo;
  }
  float x = c * (1 - std::abs(fmodu - 1));
  switch (h_category) {
    case kAdjustSaturationZero:
      rr = c;
      gg = x;
      bb = 0;
      break;
    case kAdjustSaturationOne:
      rr = x;
      gg = c;
      bb = 0;
      break;
    case kAdjustSaturationTwo:
      rr = 0;
      gg = c;
      bb = x;
      break;
    case kAdjustSaturationThree:
      rr = 0;
      gg = x;
      bb = c;
      break;
    case kAdjustSaturationFour:
      rr = x;
      gg = 0;
      bb = c;
      break;
    case kAdjustSaturationFive:
      rr = c;
      gg = 0;
      bb = x;
      break;
    default:
      rr = 0;
      gg = 0;
      bb = 0;
  }
  *r = static_cast<T>(rr + m);
  *g = static_cast<T>(gg + m);
  *b = static_cast<T>(bb + m);
}

template <typename T>
bool LaunchAdjustSaturationKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto input{static_cast<T *>(inputs[0]->addr)};
  auto scale{static_cast<std::float_t *>(inputs[1]->addr)};
  auto output{static_cast<T *>(outputs[0]->addr)};
  constexpr int64_t kChannelSize = 3;
  std::int64_t num_elements = static_cast<int64_t>(inputs[0]->size / sizeof(T));
  auto sharder_adjustsaturation = [input, scale, output, kChannelSize](int64_t start, int64_t end) {
    for (int64_t i = start * kChannelSize; i < end * kChannelSize; i = i + kChannelSize) {
      float h, s, v;
      // Convert the RGB color to Hue/V-range.
      rgb_to_hsv(static_cast<float>(*(input + i)), static_cast<float>(*(input + i + 1)),
                 static_cast<float>(*(input + i + 2)), &h, &s, &v);
      s = std::min(1.0f, std::max(0.0f, s * scale[0]));
      // Convert the hue and v-range back into RGB.
      hsv_to_rgb<T>(h, s, v, &output[i], &output[i + 1], &output[i + 2]);
    }
  };
  std::int64_t total = num_elements / kChannelSize;
  if (total > kAdjustSaturationParallelNum) {
    std::int64_t per_unit_size =
      total / std::min(kAdjustSaturationParallelNum - SizeToLong(kAdjustSaturationTwo), total);
    CPUKernelUtils::ParallelFor(sharder_adjustsaturation, static_cast<size_t>(total),
                                static_cast<float>(per_unit_size));
  } else {
    sharder_adjustsaturation(0, total);
  }
  return true;
}
}  // namespace detail

bool AdjustSaturationCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdjustSaturationTwo, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdjustSaturationOne, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  input_type_ = inputs[kIndex0]->GetDtype();
  return true;
}

bool AdjustSaturationCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &workspace,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  if (input_type_ == kNumberTypeFloat32) {
    return detail::LaunchAdjustSaturationKernel<float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeFloat16) {
    return detail::LaunchAdjustSaturationKernel<Eigen::half>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported input data type " << TypeIdLabel(input_type_);
  }
}
std::vector<KernelAttr> AdjustSaturationCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdjustSaturation, AdjustSaturationCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
