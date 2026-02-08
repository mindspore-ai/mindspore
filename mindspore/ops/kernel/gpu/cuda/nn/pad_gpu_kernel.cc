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

#include "kernel/gpu/cuda/nn/pad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/pad.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadInputsNum = 1;
constexpr size_t kPadOutputsNum = 1;
constexpr size_t kPadElemSize = 2;
}  // namespace

template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool PadFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Pad', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int PadFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  input_rank_ = input_shape_.size();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }

  auto paddings_v = primitive_->GetAttr(kAttrPaddings);
  MS_EXCEPTION_IF_NULL(paddings_v);
  std::vector<std::vector<int64_t>> paddings = GetValue<std::vector<std::vector<int64_t>>>(paddings_v);

  if (paddings.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'paddings' must be equal to the dimension of "
                      << "input, but got the length of 'paddings': " << paddings.size()
                      << " the dimension of input: " << input_rank_;
  }

  for (size_t i = 0; i < paddings.size(); i++) {
    if (paddings[i].size() != kPadElemSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of element of 'paddings' must be equal to 2, "
                        << "but got the size of paddings[" << i << "]: " << paddings[i].size();
    }
    flattened_paddings_.push_back(static_cast<int>(paddings[i][0]));
    flattened_paddings_.push_back(static_cast<int>(paddings[i][1]));
  }

  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 0; i < input_rank_; i++) {
    input_size_ *= input_shape_[i];
    output_size_ *=
      (input_shape_[i] + flattened_paddings_[kPadElemSize * i] + flattened_paddings_[(kPadElemSize * i) + 1]);
  }

  if (input_rank_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to 0, but "
                      << "got the " << input_rank_;
  }
  strides_.resize(input_rank_);
  strides_[input_rank_ - 1] = 1;
  for (int i = SizeToInt(input_rank_) - 2; i >= 0; i--) {
    strides_[i] = static_cast<int>(output_shape[i + 1]) * strides_[i + 1];
  }

  return static_cast<int>(KRET_OK);
}

template <typename T>
bool PadFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_device = GetDeviceAddress<T>(inputs, 0);
  T *output_device = GetDeviceAddress<T>(outputs, 0);

  float pad_value = 0.0;
  auto status = FillDeviceArray(output_size_, output_device, pad_value, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);

  // input_shape, strides, paddings
  PadInfo info;
  const size_t kValue2 = 2;
  for (size_t i = 0; i < input_rank_; ++i) {
    info.shape[i] = static_cast<int>(input_shape_[i]);
    info.strides[i] = strides_[i];
    info.paddings[kValue2 * i] = flattened_paddings_[kValue2 * i];
    info.paddings[kValue2 * i + 1] = flattened_paddings_[kValue2 * i + 1];
  }

  status = CalPadGeneral(input_device, output_device, info, input_size_, input_rank_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, PadFwdGpuKernelMod::PadFunc>> PadFwdGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), &PadFwdGpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &PadFwdGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &PadFwdGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &PadFwdGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &PadFwdGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &PadFwdGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &PadFwdGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &PadFwdGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &PadFwdGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PadFwdGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PadFwdGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &PadFwdGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &PadFwdGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &PadFwdGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> PadFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PadFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Pad, PadFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
