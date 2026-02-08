/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda/nn/resize_bilinear_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "kernel/gpu/cuda_impl/cuda_ops/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kResizeBilinearV2InputsNum = 4;
constexpr size_t kResizeBilinearExpectedRank = 4;
constexpr size_t kOutputsNum = 1;
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
constexpr size_t kThree = 3;
}  // namespace

using FuncVec = std::vector<std::pair<KernelAttr, ResizeBilinearGpuKernelMod::ResizeBilinearFunc>>;

bool ResizeBilinearGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kResizeBilinearV2InputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kInputsNum << ", but got "
                      << inputs.size();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'ResizeBilinearV2', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ResizeBilinearGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  if (input_shape.size() != kResizeBilinearExpectedRank || output_shape.size() != kResizeBilinearExpectedRank) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", input and output should be 4-D Tensor.";
  }
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }
  n_ = LongToInt(input_shape[kZero]);
  c_ = LongToInt(input_shape[kOne]);
  input_h_ = LongToInt(input_shape[kTwo]);
  input_w_ = LongToInt(input_shape[kThree]);
  output_h_ = LongToInt(output_shape[kTwo]);
  output_w_ = LongToInt(output_shape[kThree]);

  align_corners_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  half_pixel_centers_ = inputs[kIndex3]->GetValueWithCheck<bool>();
  return KRET_OK;
}

template <typename T>
bool ResizeBilinearGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, kZero);
  MS_EXCEPTION_IF_NULL(input);
  T *output = GetDeviceAddress<T>(outputs, kZero);
  MS_EXCEPTION_IF_NULL(output);
  float h_scale = Scaling(input_h_, output_h_, align_corners_);
  float w_scale = Scaling(input_w_, output_w_, align_corners_);
  auto status = CalResizeBilinear(input, n_, c_, input_h_, input_w_, output_h_, output_w_, h_scale, w_scale,
                                  half_pixel_centers_, output, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

FuncVec ResizeBilinearGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> ResizeBilinearGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBilinearFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeBilinearV2, ResizeBilinearGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
