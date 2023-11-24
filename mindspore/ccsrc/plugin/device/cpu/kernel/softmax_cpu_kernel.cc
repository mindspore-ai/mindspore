/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/softmax_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/softmax_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSoftmaxAxisNum = 1;
}  // namespace

bool SoftmaxCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Softmax does not support this kernel data type: " << kernel_attr;
  }

  return true;
}

int SoftmaxCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto axis_list = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (axis_list.size() != kSoftmaxAxisNum) {
    MS_LOG(EXCEPTION) << "For Softmax, the parameter 'axis' only support int type on CPU, but got tuple.";
  }
  axis_ = static_cast<int32_t>(axis_list[0]);
  dtype_ = inputs[0]->dtype_id();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  size_t shape_size = input_shape.size();
  if (shape_size == 0) {
    MS_LOG(ERROR) << "Input shape size is 0.";
    return KRET_RESIZE_FAILED;
  }
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  input_dims_ = SizeToInt(shape_size);
  if (axis_ < -input_dims_ || axis_ >= input_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in range [" << -input_dims_ << ", "
                      << input_dims_ << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += input_dims_;
  }
  auto input_elements = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  channel_ = input_shape_[axis_];
  output_elements_ = input_elements / static_cast<size_t>(channel_);
  if (axis_ == input_dims_ - 1) {
    last_axis_ = true;
  }
  const auto dtype_size = abstract::TypeIdSize(dtype_);
  workspace_size_list_.clear();
  workspace_size_list_ = {output_elements_ * dtype_size};
  return KRET_OK;
}

void SoftmaxCpuKernelMod::LaunchKernelLastAxis(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  float *input_data = GetDeviceAddress<float>(inputs, kIndex0);
  float *output_data = GetDeviceAddress<float>(outputs, kIndex0);
  auto task = [this, input_data, output_data](size_t start, size_t end) {
    int batch = SizeToInt(end - start);
    size_t offset = start * IntToSize(channel_);
    (void)SoftmaxLastAxis(input_data + offset, output_data + offset, batch, channel_);
  };
  ParallelLaunchAutoSearch(task, output_elements_, this, &parallel_search_info_);
}

void SoftmaxCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs) {
  float *input_data = GetDeviceAddress<float>(inputs, kIndex0);
  float *output_data = GetDeviceAddress<float>(outputs, kIndex0);
  float *sum_data = GetDeviceAddress<float>(workspace, kIndex0);
  auto task = [this, input_data, output_data, sum_data](size_t start, size_t end) {
    (void)Softmax(input_data, output_data, sum_data, axis_, input_dims_, &input_shape_[0]);
  };
  ParallelLaunchAutoSearch(task, output_elements_, this, &parallel_search_info_);
}

bool SoftmaxCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    if (last_axis_) {
      LaunchKernelLastAxis(inputs, outputs);
    } else {
      LaunchKernel(inputs, workspace, outputs);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' should be float32, "
                         "but got "
                      << TypeIdLabel(dtype_);
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Softmax, SoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
