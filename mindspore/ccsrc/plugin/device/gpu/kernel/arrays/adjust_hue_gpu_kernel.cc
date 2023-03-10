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
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/arrays/adjust_hue_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adjusthue_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int64_t INPUT_IMAGES_MIN_DIMS = 3;
constexpr int64_t INPUT_IMAGES_LAST_DIM = 3;
constexpr int64_t DETLA_DIMS = 0;
constexpr size_t INPUT_BUM = 2;
constexpr size_t OUTPUT_NUM = 1;

void AdjustHueGpuKernelMod::ResetResource() {
  stream_ptr_ = nullptr;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
}

void AdjustHueGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_elements * data_unit_size_);
  input_size_list_.push_back(sizeof(float));
  output_size_list_.push_back(input_elements * data_unit_size_);
}

bool AdjustHueGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_BUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "'get empty inputs or outputs, which is invalid.";
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

int AdjustHueGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  std::vector<size_t> shape = std::vector<size_t>(inputs[kIndex0]->GetDeviceShapeAdaptively().begin(),
                                                  inputs[kIndex0]->GetDeviceShapeAdaptively().end());
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (!is_null_input_) {
    for (size_t i = 0; i < shape.size(); ++i) {
      input_elements *= shape[i];
    }
  }
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  int64_t input_dims = shape.size();
  if (input_dims < INPUT_IMAGES_MIN_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_images' should be equal to "
                     "3-D or greater than 3-D, but got "
                  << input_dims << "-D.";
    return false;
  }
  if (shape[input_dims - 1] != INPUT_IMAGES_LAST_DIM) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the last dimension of 'input_images' should be equal "
                     "to 3, but got "
                  << shape[input_dims - 1] << ".";
    return false;
  }
  std::vector<size_t> delta_shape = std::vector<size_t>(inputs[kIndex1]->GetDeviceShapeAdaptively().begin(),
                                                        inputs[kIndex1]->GetDeviceShapeAdaptively().end());
  int64_t delta_dims = delta_shape.size();
  if (delta_dims != DETLA_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the  dimension of 'dalta' should be equal to 0-D, but got "
                  << delta_dims << "-D.";
    return false;
  }
  ResetResource();
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool AdjustHueGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  stream_ptr_ = stream_ptr;
  T *input_image = GetDeviceAddress<T>(inputs, kIndex0);
  float *hue_delta = GetDeviceAddress<float>(inputs, kIndex1);
  T *output_image = GetDeviceAddress<T>(outputs, kIndex0);
  auto status = CalAdjusthue(input_elements, input_image, output_image, hue_delta, device_id_,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, AdjustHueGpuKernelMod::AdjustHueFunc>> AdjustHueGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
   &AdjustHueGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &AdjustHueGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
   &AdjustHueGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> AdjustHueGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdjustHueFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdjustHue, AdjustHueGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
