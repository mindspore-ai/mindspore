/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/resize_bilinear_grad_cpu_kernel.h"
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeBilinearGradInputsNum = 2;
constexpr size_t kResizeBilinearGradOutputNum = 1;
constexpr size_t kResizeBilinearGradInputsDoutShapeSize = 4;
constexpr size_t kResizeBilinearGradInputsXShapeSize = 4;
}  // namespace

using FuncVec = const std::vector<std::pair<KernelAttr, ResizeBilinearGradCpuKernelMod::KernelRunFunc>>;

bool ResizeBilinearGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kResizeBilinearGradInputsNum || outputs.size() != kResizeBilinearGradOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be "
                  << kResizeBilinearGradInputsNum << " and " << kResizeBilinearGradOutputNum << ", but got "
                  << inputs.size() << " and " << outputs.size();
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ResizeBilinearGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  size_ = Convert2SizeTClipNeg(inputs[kIndex1]->GetShapeVector());
  align_corners_ = GetValue<bool>(base_operator->GetAttr(kAttrAlignCorners));
  is_null_input_ = (std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>()) == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }

  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  height_scale = Scaling(out_height, in_height, align_corners_);
  width_scale = Scaling(out_width, in_width, align_corners_);
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool ResizeBilinearGradCpuKernelMod::LaunchFloat16Kernel(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto *output_addr = reinterpret_cast<float16 *>(outputs[0]->addr);
  if (memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  auto *input_addr_T = reinterpret_cast<float16 *>(inputs[0]->addr);
  size_t input_mem_size = inputs[0]->size / sizeof(float16) * sizeof(float);
  float *float_dloss_addr = reinterpret_cast<float *>(malloc(input_mem_size));
  if (float_dloss_addr == NULL) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc memory failed.";
    return false;
  }
  for (size_t i = 0; i < ((inputs[0]->size) / sizeof(float16)); ++i) {
    float_dloss_addr[i] = static_cast<float>(input_addr_T[i]);
  }

  size_t output_mem_size = outputs[0]->size / sizeof(float16) * sizeof(float);
  float *float_output_addr = reinterpret_cast<float *>(malloc(output_mem_size));
  if (float_output_addr == NULL) {
    free(float_dloss_addr);
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc memory failed.";
    return false;
  }
  size_t memset_size = outputs[0]->size / sizeof(float16) * sizeof(float);
  if (memset_s(float_output_addr, memset_size, 0, memset_size) != EOK) {
    free(float_dloss_addr);
    free(float_output_addr);
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(float_dloss_addr);
  MS_EXCEPTION_IF_NULL(float_output_addr);

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;

  float *cur_dloss_addr = float_dloss_addr;
  float *cur_output_addr = float_output_addr;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < in_height; ++h) {
        const float in_y = static_cast<float>(h) * height_scale;
        const size_t top_y_index = std::max(static_cast<size_t>(floorf(in_y)), static_cast<size_t>(0));
        const size_t bottom_y_index = std::min(static_cast<size_t>(ceilf(in_y)), out_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = 1.0 - y_lerp;
        for (size_t w = 0; w < in_width; ++w) {
          const float in_x = static_cast<float>(w) * width_scale;
          const size_t left_x_index = std::max(static_cast<size_t>(floorf(in_x)), static_cast<size_t>(0));
          const size_t right_x_index = std::min(static_cast<size_t>(ceilf(in_x)), out_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = 1.0 - x_lerp;
          cur_output_addr[top_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(inverse_y_lerp * inverse_x_lerp);
          cur_output_addr[top_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(inverse_y_lerp * x_lerp);
          cur_output_addr[bottom_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(y_lerp * inverse_x_lerp);
          cur_output_addr[bottom_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(y_lerp * x_lerp);

          output_addr[top_y_index * out_width + left_x_index] =
            static_cast<float16>(cur_output_addr[top_y_index * out_width + left_x_index]);
          output_addr[top_y_index * out_width + right_x_index] =
            static_cast<float16>(cur_output_addr[top_y_index * out_width + right_x_index]);
          output_addr[bottom_y_index * out_width + left_x_index] =
            static_cast<float16>(cur_output_addr[bottom_y_index * out_width + left_x_index]);
          output_addr[bottom_y_index * out_width + right_x_index] =
            static_cast<float16>(cur_output_addr[bottom_y_index * out_width + right_x_index]);
        }
      }
      output_addr += out_hw_size;
      cur_dloss_addr += in_hw_size;
      cur_output_addr += out_hw_size;
    }
  }
  free(float_dloss_addr);
  free(float_output_addr);
  return true;
}

template <typename T>
bool ResizeBilinearGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  auto *float_dloss_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *float_output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(float_dloss_addr);
  MS_EXCEPTION_IF_NULL(float_output_addr);

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;

  T *cur_dloss_addr = float_dloss_addr;
  T *cur_output_addr = float_output_addr;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < in_height; ++h) {
        const T in_y = static_cast<T>(h) * height_scale;
        const size_t top_y_index = std::max(static_cast<size_t>(floorf(in_y)), static_cast<size_t>(0));
        const size_t bottom_y_index = std::min(static_cast<size_t>(ceilf(in_y)), out_height - 1);
        const T y_lerp = in_y - floorf(in_y);
        const T inverse_y_lerp = 1.0 - y_lerp;
        for (size_t w = 0; w < in_width; ++w) {
          const T in_x = static_cast<T>(w) * width_scale;
          const size_t left_x_index = std::max(static_cast<size_t>(floorf(in_x)), static_cast<size_t>(0));
          const size_t right_x_index = std::min(static_cast<size_t>(ceilf(in_x)), out_width - 1);
          const T x_lerp = in_x - floorf(in_x);
          const T inverse_x_lerp = 1.0 - x_lerp;
          cur_output_addr[top_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<T>(inverse_y_lerp * inverse_x_lerp);
          cur_output_addr[top_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<T>(inverse_y_lerp * x_lerp);
          cur_output_addr[bottom_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<T>(y_lerp * inverse_x_lerp);
          cur_output_addr[bottom_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<T>(y_lerp * x_lerp);

          output_addr[top_y_index * out_width + left_x_index] =
            static_cast<T>(cur_output_addr[top_y_index * out_width + left_x_index]);
          output_addr[top_y_index * out_width + right_x_index] =
            static_cast<T>(cur_output_addr[top_y_index * out_width + right_x_index]);
          output_addr[bottom_y_index * out_width + left_x_index] =
            static_cast<T>(cur_output_addr[bottom_y_index * out_width + left_x_index]);
          output_addr[bottom_y_index * out_width + right_x_index] =
            static_cast<T>(cur_output_addr[bottom_y_index * out_width + right_x_index]);
        }
      }
      output_addr += out_hw_size;
      cur_dloss_addr += in_hw_size;
      cur_output_addr += out_hw_size;
    }
  }
  return true;
}

FuncVec &ResizeBilinearGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeBilinearGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ResizeBilinearGradCpuKernelMod::LaunchFloat16Kernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ResizeBilinearGradCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ResizeBilinearGradCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeBilinearGrad, ResizeBilinearGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
