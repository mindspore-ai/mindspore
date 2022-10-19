/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/extract_glimpse_cpu_kernel.h"
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <utility>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/extract_glimpse.h"

namespace {
const size_t kDataSizeThreshold = 4 * 1024;
using std::random_device;
const size_t kNumber10 = 10;
const float kNumber11 = 0.5;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis_uniform(-1, 1);
std::normal_distribution<float> dis_normal(kNumber10, kNumber11);
}  // namespace

namespace mindspore {
namespace kernel {
bool ExtractGlimpseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ExtractGlimpse>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  centered_ = kernel_ptr->get_centered();
  normalized_ = kernel_ptr->get_normalized();
  uniform_noise_ = kernel_ptr->get_uniform_noise();
  noise_ = kernel_ptr->get_noise();
  return true;
}

int ExtractGlimpseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_dtype_ = inputs[kIndex0]->GetDtype();
  size_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  offsets_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

bool ExtractGlimpseCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  bool ret = true;
  if (input_dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]:" << TypeIdToType(input_dtype_)->ToString();
  }
  return ret;
}

void Necessity(uint64_t un, bool u_n, float *y_d, uint64_t p_y, string no) {
  if (u_n) {
    y_d[p_y + un] = dis_uniform(gen);
  } else if (no == "zero") {
    y_d[p_y + un] = 0.0f;
  } else if (no == "gaussian") {
    y_d[p_y + un] = std::max(0.0f, dis_normal(gen));
  } else if (no == "uniform") {
    y_d[p_y + un] = dis_uniform(gen);
  } else {
    MS_LOG(EXCEPTION) << "noise type unsupported.";
  }
}

std::pair<float, float> ExtractGlimpseCpuKernelMod::GetLocation(const float *ptr, const uint64_t seq,
                                                                const std::pair<uint64_t, uint64_t> image_size,
                                                                const std::pair<uint64_t, uint64_t> g_size,
                                                                const bool normalized, const bool centered) {
  float x = ptr[seq << 1];
  float y = ptr[1 + (seq << 1)];
  uint64_t image_height = image_size.first;
  uint64_t image_width = image_size.second;
  uint64_t g_height = g_size.first;
  uint64_t g_width = g_size.second;
  if (normalized) {
    x *= static_cast<float>(image_height);
    y *= static_cast<float>(image_width);
  }
  if (centered) {
    x /= 2.0f;
    y /= 2.0f;
    x += image_height / 2.0f;
    y += image_width / 2.0f;
  }
  x -= g_height / 2.0f;
  y -= g_width / 2.0f;
  return std::pair<float, float>(x, y);
}

template <typename T>
bool ExtractGlimpseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  const size_t kInputIndex3 = 2;
  const size_t kInputIndex4 = 3;
  const size_t kNumber8 = 8;
  const size_t kNumber1024 = 1024;
  float *x_data = static_cast<float *>(inputs[0]->addr);
  int32_t *ss_data = static_cast<int32_t *>(inputs[1]->addr);
  float *offsets_data = static_cast<float *>(inputs[kInputIndex3]->addr);
  float *y_data = static_cast<float *>(outputs[0]->addr);
  uint64_t batch_cnt = static_cast<uint64_t>(input_shape_[0]);
  uint64_t image_height = static_cast<uint64_t>(input_shape_[1]);
  uint64_t image_width = static_cast<uint64_t>(input_shape_[kInputIndex3]);
  uint64_t channels = static_cast<uint64_t>(input_shape_[kInputIndex4]);
  uint64_t g_height = static_cast<uint64_t>(ss_data[0]), g_width = static_cast<uint64_t>(ss_data[1]);
  uint64_t size1 = image_width * image_height * channels;
  uint64_t size2 = image_width * channels;
  uint64_t size3 = g_height * g_width * channels;
  uint64_t size4 = size3 / g_height;
  uint64_t g_size = g_width * g_height;
  std::pair<uint64_t, uint64_t> image_size(image_height, image_width);
  std::pair<uint64_t, uint64_t> win_size(g_height, g_width);
  if (batch_cnt < kNumber8 * kNumber1024) {
    for (uint64_t i = 0; i < batch_cnt; i++) {
      std::pair<float, float> loc = GetLocation(offsets_data, i, image_size, win_size, normalized_, centered_);
      float x = loc.first;
      float y = loc.second;
      auto task = [&](int64_t st, int64_t ed) {
        for (int64_t v = st; v < ed; v++) {
          int64_t j = v / static_cast<int64_t>(g_width), k = v % static_cast<int64_t>(g_width);
          uint64_t a = static_cast<uint64_t>(FloatToLong(x) + j), b = static_cast<uint64_t>(FloatToLong(y) + k);
          uint64_t pos_y = i * size3 + static_cast<int64_t>(j) * size4 + static_cast<int64_t>(k) * channels;
          if (a >= image_height || b >= image_width) {
            for (uint64_t u = 0; u < channels; u++) {
              Necessity(u, uniform_noise_, y_data, pos_y, noise_);
            }
            continue;
          }
          uint64_t pos_x = i * size1 + a * size2 + b * channels;
          for (uint64_t u = 0; u < channels; u++) {
            y_data[pos_y + u] = x_data[pos_x + u];
          }
        }
      };
      if (g_size < kNumber8 * kNumber1024) {
        task(0, static_cast<int64_t>(g_size));
      } else {
        CPUKernelUtils::ParallelFor(task, g_size);
      }
    }
  } else {
    auto task = [&](size_t st, size_t ed) {
      for (uint64_t i = st; i < ed; i++) {
        std::pair<float, float> loc = GetLocation(offsets_data, i, image_size, win_size, normalized_, centered_);
        float x = loc.first;
        float y = loc.second;
        for (uint64_t v = 0; v < g_size; v++) {
          int64_t j = static_cast<int64_t>(v / g_width), k = static_cast<int64_t>(v % g_width);
          uint64_t a = static_cast<uint64_t>(FloatToLong(x) + j), b = static_cast<uint64_t>(FloatToLong(y) + k);
          uint64_t pos_y = i * size3 + static_cast<int64_t>(j) * size4 + static_cast<int64_t>(k) * channels;
          if (a >= image_height || b >= image_width) {
            for (uint64_t u = 0; u < channels; u++) {
              Necessity(u, uniform_noise_, y_data, pos_y, noise_);
            }
            continue;
          }
          uint64_t pos_x = i * size1 + a * size2 + b * channels;
          for (uint64_t u = 0; u < channels; u++) {
            y_data[pos_y + u] = x_data[pos_x + u];
          }
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, batch_cnt);
  }
  return true;
}

std::vector<KernelAttr> ExtractGlimpseCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ExtractGlimpse, ExtractGlimpseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
