/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/resize_bilinear.h"
#include "plugin/device/gpu/kernel/nn/resize_bilinear_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kDynamicInputsNum = 2;
constexpr size_t kOutputsNum = 1;
constexpr size_t kImagRank = 4;
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
constexpr size_t kThree = 3;
}  // namespace

using FuncVec = std::vector<std::pair<KernelAttr, ResizeBilinearGpuKernelMod::ResizeBilinearFunc>>;

bool ResizeBilinearGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputsNum && inputs.size() != kDynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1 or 2"
                      << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'ResizeBilinear', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ResizeBilinearGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }
  if (input_shape.size() != kImagRank || output_shape.size() != kImagRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output must be equal to 4, but "
                      << "got the dimension of input: " << input_shape.size()
                      << ", the dimension of output: " << output_shape.size();
  }
  n_ = LongToInt(input_shape[kZero]);
  c_ = LongToInt(input_shape[kOne]);
  input_h_ = LongToInt(input_shape[kTwo]);
  input_w_ = LongToInt(input_shape[kThree]);
  output_h_ = LongToInt(output_shape[kTwo]);
  output_w_ = LongToInt(output_shape[kThree]);
  input_size_ = abstract::TypeIdSize(inputs[kZero]->GetDtype()) * SizeOf(input_shape);
  output_size_ = abstract::TypeIdSize(outputs[kZero]->GetDtype()) * SizeOf(output_shape);
  auto align_corners = base_operator->GetAttr(kAttrAlignCorners);
  MS_EXCEPTION_IF_NULL(align_corners);
  align_corners_ = GetValue<bool>(align_corners);
  auto half_pixel_centers = base_operator->GetAttr(kAttrHalfPixelCenters);
  MS_EXCEPTION_IF_NULL(half_pixel_centers);
  half_pixel_centers_ = GetValue<bool>(half_pixel_centers);
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool ResizeBilinearGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, kZero);
  T *output = GetDeviceAddress<T>(outputs, kZero);
  float h_scale = Scaling(input_h_, output_h_, align_corners_);
  float w_scale = Scaling(input_w_, output_w_, align_corners_);
  CalResizeBilinear(input, n_, c_, input_h_, input_w_, output_h_, output_w_, h_scale, w_scale, half_pixel_centers_,
                    output, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

FuncVec ResizeBilinearGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> ResizeBilinearGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBilinearFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeBilinear, ResizeBilinearGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeBilinearV2, ResizeBilinearGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
