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

#include "plugin/device/cpu/kernel/histogram_cpu_kernel.h"
#include <algorithm>
#include <ctime>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include "kernel/common_utils.h"
#include "mindspore/core/ops/histogram.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
}  // namespace

bool HistogramCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  x_type_ = inputs[kIndex0]->GetDtype();
  auto op_prim = std::dynamic_pointer_cast<ops::Histogram>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  bins_ = op_prim->get_bins();
  min_attr_ = op_prim->get_min();
  max_attr_ = op_prim->get_max();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

bool HistogramCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  // Benchmarking framework only support float32 and float64
  // To meet precision requirements, cast float16 or int32 to float32
  switch (x_type_) {
    case kNumberTypeFloat16: {
      LaunchKernel<float16, float>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat32: {
      LaunchKernel<float, float>(inputs, outputs);
      break;
    }
    case kNumberTypeInt32: {
      LaunchKernel<int32_t, float>(inputs, outputs);
      break;
    }
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the dtype of 'x' should be float16, float32, int32, but got "
                              << TypeIdLabel(x_type_);
  }
  return true;
}

template <typename T, typename InterType>
void HistogramCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  if (inputs[kIndex0]->size == 0) {
    return;
  }
  auto x_data = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto y_data = reinterpret_cast<int32_t *>(outputs[kIndex0]->addr);
  size_t x_num = inputs[kIndex0]->size / sizeof(T);
  const int32_t y_num = LongToInt(bins_);
  // initial y as all zero
  std::fill(y_data, y_data + y_num, 0);
  // calculate left and right of input
  double leftmost_edge = static_cast<double>(min_attr_);
  double rightmost_edge = static_cast<double>(max_attr_);
  if (leftmost_edge > rightmost_edge) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', max must be larger than min. bu get attr min=" << leftmost_edge
                             << ", attr max =" << rightmost_edge << ".";
  }
  auto min_max = std::minmax_element(x_data, x_data + x_num);
  auto x_min = static_cast<double>(*min_max.first);
  auto x_max = static_cast<double>(*min_max.second);
  if (leftmost_edge == rightmost_edge && x_num > 0) {
    leftmost_edge = x_min;
    rightmost_edge = x_max;
  } else if (x_min > rightmost_edge || x_max < leftmost_edge) {
    return;
  }
  if (leftmost_edge == rightmost_edge) {
    leftmost_edge -= 1;
    rightmost_edge += 1;
  }
  if (std::isinf(leftmost_edge) || std::isinf(rightmost_edge) || std::isnan(leftmost_edge) ||
      std::isnan(rightmost_edge)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', range of [" << leftmost_edge << ", " << rightmost_edge
                             << "] is not finite.";
  }

  const InterType step = static_cast<InterType>(rightmost_edge) - static_cast<InterType>(leftmost_edge);
  const int64_t nbins_minus_1 = bins_ - 1;
  std::mutex hist_mutex;
  auto sharder_histogram = [&](int64_t start, int64_t end) {
    std::vector<int32_t> hist_local(y_num, 0);
    for (int64_t i = start; i < end; ++i) {
      auto elt = static_cast<InterType>(x_data[i]);
      if (elt < static_cast<InterType>(leftmost_edge) || elt > static_cast<InterType>(rightmost_edge)) {
        continue;
      }
      int64_t pos =
        static_cast<int64_t>((elt - static_cast<InterType>(leftmost_edge)) / step * static_cast<InterType>(bins_));
      pos = std::min(pos, nbins_minus_1);
      hist_local[pos] += static_cast<int32_t>(1);
    }
    // Locks and updates the common output
    const std::lock_guard<std::mutex> lock(hist_mutex);
    (void)std::transform(hist_local.begin(), hist_local.end(), y_data, y_data, std::plus<int32_t>());
  };
  CPUKernelUtils::ParallelFor(sharder_histogram, x_num);
}

std::vector<KernelAttr> HistogramCPUKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Histogram, HistogramCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
