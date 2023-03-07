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

#include "plugin/device/gpu/kernel/arrays/diag_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/diag_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kDiagInputsNum = 1;
constexpr int kDiagOutputsNum = 1;
}  // namespace

std::vector<std::pair<KernelAttr, DiagGpuKernelMod::DiagLaunchFunc>> DiagGpuKernelMod::diag_func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), &DiagGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &DiagGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &DiagGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &DiagGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &DiagGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &DiagGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &DiagGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &DiagGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &DiagGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &DiagGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &DiagGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &DiagGpuKernelMod::LaunchKernel<utils::Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &DiagGpuKernelMod::LaunchKernel<utils::Complex<double>>}};

std::vector<KernelAttr> DiagGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(diag_func_list_.begin(), diag_func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DiagGpuKernelMod::DiagLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool DiagGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDiagOutputsNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = diag_func_list_[index].second;
  batch_rank_ = base_operator->get_batch_rank();
  return true;
}

int DiagGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  // Get the input size of each batch.
  auto input = inputs.at(kIndex0);
  MS_EXCEPTION_IF_NULL(input);
  auto input_shape = input->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL((input_shape.size() > LongToSize(batch_rank_)),
                             "The input shape should be larger than batch rank.");
  batch_size_ = 1;
  for (size_t i = 0; i < LongToSize(batch_rank_); ++i) {
    batch_size_ *= LongToSize(input_shape[i]);
  }
  input_size_ = 1;
  for (size_t i = LongToSize(batch_rank_); i < input_shape.size(); ++i) {
    input_size_ *= LongToSize(input_shape[i]);
  }
  if (input_size_ == 0) {
    MS_LOG(ERROR) << kernel_name_ << "input size should should be larger than 0, but got: " << input_size_;
    return KRET_RESIZE_FAILED;
  }

  // Get the output size of each batch.
  auto output = outputs.at(kIndex0);
  MS_EXCEPTION_IF_NULL(output);
  auto output_shape = output->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL((output_shape.size() > LongToSize(batch_rank_)),
                             "The output shape should be larger than batch rank.");
  output_size_ = 1;
  for (size_t i = LongToSize(batch_rank_); i < output_shape.size(); ++i) {
    output_size_ *= LongToSize(output_shape[i]);
  }

  if (output_size_ / input_size_ != input_size_) {
    MS_LOG(ERROR) << kernel_name_ << " resize failed, input size: " << input_size_ << ", output size: " << output_size_
                  << ", batch size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
  MS_LOG(DEBUG) << kernel_name_ << " input size: " << input_size_ << ", output size: " << output_size_
                << ", batch size: " << batch_size_;
  return KRET_OK;
}

template <typename DataType>
bool DiagGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input_begin_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(input_begin_ptr);
  auto output_begin_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output_begin_ptr);

  // Support the batch calculation of vmap.
  for (size_t i = 0; i < batch_size_; ++i) {
    auto input_ptr = input_begin_ptr + i * input_size_;
    auto output_ptr = output_begin_ptr + i * output_size_;
    CalDiag(input_ptr, output_ptr, input_size_, output_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Diag, DiagGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
