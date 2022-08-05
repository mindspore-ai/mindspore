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

#include "plugin/device/gpu/kernel/nn/pad_gpu_kernel.h"
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/pad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadInputsNum = 1;
constexpr size_t kPadOutputsNum = 1;
constexpr size_t kPadElemSize = 2;
}  // namespace

template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool PadFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Pad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Pad ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
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

int PadFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  input_rank_ = input_shape_.size();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }

  auto paddings_v = base_operator->GetAttr(kAttrPaddings);
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
    flattened_paddings_.push_back(static_cast<int32_t>(paddings[i][0]));
    flattened_paddings_.push_back(static_cast<int32_t>(paddings[i][1]));
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
  if (output_shape.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output must be the same, but "
                      << "got the dimension of input: " << input_rank_
                      << ", the dimension of output: " << output_shape.size();
  }
  strides_.resize(input_rank_);
  strides_[input_rank_ - 1] = 1;
  for (int i = SizeToInt(input_rank_) - 2; i >= 0; i--) {
    strides_[i] = static_cast<size_t>(output_shape[i + 1]) * strides_[i + 1];
  }
  InitWorkspaceSizeLists();
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool PadFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_device = GetDeviceAddress<T>(inputs, 0);
  T *output_device = GetDeviceAddress<T>(outputs, 0);

  float pad_value = 0.0;
  FillDeviceArray(output_size_, output_device, pad_value, reinterpret_cast<cudaStream_t>(stream_ptr));

  size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, kIndex0);
  size_t *strides_device = GetDeviceAddress<size_t>(workspace, kIndex1);
  int32_t *paddings_device = GetDeviceAddress<int32_t>(workspace, kIndex2);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[kIndex0], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "For 'Pad', cudaMemcpyAsync for input_shape_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(strides_device, &strides_[0], workspace_size_list_[kIndex1], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "For 'Pad', cudaMemcpyAsync for strides_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(paddings_device, &flattened_paddings_[0], workspace_size_list_[kIndex2], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "For 'Pad', cudaMemcpyAsync for paddings_ failed");

  CalPadGeneral(input_device, output_device, input_shape_device, strides_device, paddings_device, input_size_,
                input_rank_, reinterpret_cast<cudaStream_t>(stream_ptr));

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
