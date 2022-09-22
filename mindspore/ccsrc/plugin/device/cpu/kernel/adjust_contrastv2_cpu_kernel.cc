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

#include "plugin/device/cpu/kernel/adjust_contrastv2_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdjustContrastv2InputNum = 2;
constexpr size_t kAdjustContrastv2OutputNum = 1;
const int64_t kAdjustContrastv2ParallelNum = 64 * 1024;
}  // namespace

template <typename T>
void AdjustContrastv2(const T *image, T *image_out, std::float_t contrast_factor, std::int64_t channel_count,
                      std::int64_t per_batch_elements) {
  if (channel_count == 0) {
    return;
  }
  for (std::int64_t j{0}; j < channel_count; j++) {
    std::float_t sum{0.0f};
    for (std::int64_t i{0}; i < per_batch_elements; i += channel_count) {
      sum += static_cast<std::float_t>(image[i + j]);
    }
    std::float_t mean{sum / (per_batch_elements / channel_count)};
    for (std::int64_t i{0}; i < per_batch_elements; i += channel_count) {
      image_out[i + j] = static_cast<T>((static_cast<std::float_t>(image[i + j]) - mean) * contrast_factor + mean);
    }
  }
}

bool ParallelForAdjustContrastv2(std::int64_t total, std::int64_t per_unit_size,
                                 const std::function<void(std::int64_t, std::int64_t)> work) {
  if (total > kAdjustContrastv2ParallelNum)
    CPUKernelUtils::ParallelFor(work, static_cast<size_t>(total), static_cast<float>(per_unit_size));
  else
    work(0, total);
  return true;
}

template <typename T>
bool AdjustContrastv2CpuKernelMod::LaunchAdjustContrastv2Kernel(const std::vector<AddressPtr> &inputs,
                                                                const std::vector<AddressPtr> &outputs) {
  T *input{static_cast<T *>(inputs[0]->addr)};
  std::float_t *contrast_factor{static_cast<std::float_t *>(inputs[1]->addr)};
  T *output{static_cast<T *>(outputs[0]->addr)};
  std::vector<int64_t> x_dim_sizes = images_shape;
  std::size_t n{x_dim_sizes.size()};
  std::size_t per_batch_elements{LongToSize(x_dim_sizes[n - 1] * x_dim_sizes[n - 2] * x_dim_sizes[n - 3])};
  MS_EXCEPTION_IF_ZERO("per_batch_elements", per_batch_elements);
  std::int64_t input_numelements = static_cast<int64_t>(inputs[0]->size / sizeof(T));
  std::int64_t total{input_numelements / SizeToLong(per_batch_elements)};
  std::int64_t per_unit_size{total / std::min(kAdjustContrastv2ParallelNum - 2L, total)};
  return ParallelForAdjustContrastv2(total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; i += 1) {
      AdjustContrastv2(&(input[i * static_cast<int64_t>(per_batch_elements)]),
                       &(output[i * static_cast<int64_t>(per_batch_elements)]), contrast_factor[0], x_dim_sizes[n - 1],
                       per_batch_elements);
    }
  });
}

void AdjustContrastv2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<int64_t> output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  images_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (AnfAlgo::IsShapesDynamic({output_shape, images_shape})) {
    return;
  }
  if (images_shape != output_shape) {
    MS_LOG(EXCEPTION) << "For AdjustContrastv2, the data type of the input " << images_shape
                      << "need be the same as the output " << output_shape << ".";
  }
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kAdjustContrastv2InputNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kAdjustContrastv2OutputNum, kernel_name_);
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool AdjustContrastv2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &workspace,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  if (input_type_ == kNumberTypeFloat32) {
    return LaunchAdjustContrastv2Kernel<float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeFloat16) {
    return LaunchAdjustContrastv2Kernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the type of 'image' should be float16, float32, but got "
                      << TypeIdLabel(input_type_);
  }
}

std::vector<KernelAttr> AdjustContrastv2CpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdjustContrastv2, AdjustContrastv2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
