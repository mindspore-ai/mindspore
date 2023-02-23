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

#include <algorithm>
#include <map>
#include <utility>
#include <functional>

#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/arrays/adjust_contrast_v2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adjust_contrast_v2_impl.cuh"
namespace mindspore {
namespace kernel {
#define ADJUST_CONTRAST_V2_GPU_REGISTER(T_DT, T)                                        \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(T_DT), \
    &AdjustContrastV2GpuKernelMod::LaunchKernel<T>

constexpr int64_t INPUT_DIMS_3 = 3;
constexpr int64_t INPUT_DIMS_0 = 0;
constexpr int last_dim = 1, second_dim = 2, third_dim = 3;

void AdjustContrastV2GpuKernelMod::ResetResource() {
  stream_ptr_ = nullptr;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
}

void AdjustContrastV2GpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(total_ * per_batch_elements_ * data_unit_size_);
  input_size_list_.push_back(sizeof(float));

  output_size_list_.push_back(total_ * per_batch_elements_ * data_unit_size_);
}

bool AdjustContrastV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  return true;
}

int AdjustContrastV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<size_t> shape = std::vector<size_t>(inputs[kIndex0]->GetDeviceShapeAdaptively().begin(),
                                                  inputs[kIndex0]->GetDeviceShapeAdaptively().end());
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (!is_null_input_) {
    int num_dims = shape.size();
    per_batch_elements_ = shape[num_dims - last_dim] * shape[num_dims - second_dim] * shape[num_dims - third_dim];
    total_ = std::accumulate(shape.begin(), shape.end() - third_dim, third_dim, std::multiplies<int>());
    std::vector<int64_t> images_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                             inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
    std::vector<int64_t> contrast_factor_shape = std::vector<int64_t>(
      inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(), inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
    int64_t images_dims = images_shape.size();
    int64_t contrast_factor_dims = contrast_factor_shape.size();
    if (images_dims < INPUT_DIMS_3) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'images' must greater then 3, but got "
                    << images_dims << ".";
      return KRET_RESIZE_FAILED;
    }
    if (images_shape[images_dims - 1] != INPUT_DIMS_3) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the last dimension of 'images' must be 3, but got " << images_dims
                    << ".";
      return KRET_RESIZE_FAILED;
    }
    if (contrast_factor_dims != INPUT_DIMS_0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'contrast_factor' should be 0-D, but got "
                    << contrast_factor_dims << "-D.";
      return KRET_RESIZE_FAILED;
    }
    InitSizeLists();
  }
  return KRET_OK;
}

template <typename T>
bool AdjustContrastV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  stream_ptr_ = stream_ptr;
  T *images = GetDeviceAddress<T>(inputs, kIndex0);
  float *contrast_factor = GetDeviceAddress<float>(inputs, kIndex1);
  T *images_out = GetDeviceAddress<T>(outputs, kIndex0);

  auto status = CalAdjustContrastV2GpuKernel(images, contrast_factor, images_out, total_, per_batch_elements_,
                                             device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, AdjustContrastV2GpuKernelMod::AdjustContrastv2Func>>
  AdjustContrastV2GpuKernelMod::func_list_ = {{ADJUST_CONTRAST_V2_GPU_REGISTER(kNumberTypeFloat16, half)},
                                              {ADJUST_CONTRAST_V2_GPU_REGISTER(kNumberTypeFloat32, float)}};

std::vector<KernelAttr> AdjustContrastV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdjustContrastv2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdjustContrastv2, AdjustContrastV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
