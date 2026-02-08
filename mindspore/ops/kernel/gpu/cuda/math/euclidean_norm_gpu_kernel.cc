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

#include "kernel/gpu/cuda/math/euclidean_norm_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <set>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/infer/euclidean_norm.h"
#include "kernel/gpu/cuda_impl/cuda_ops/euclidean_norm_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;

void EuclideanNormGpuKernelMod::InitWorkSpaceSizeList() {
  const size_t device_input_shape_size = input_shape_.size() * sizeof(size_t);
  const size_t device_axes_shape_size = output_axes_.size() * sizeof(size_t);
  const size_t device_output_stride_size = output_stride_.size() * sizeof(size_t);

  workspace_size_list_.clear();
  workspace_size_list_ = {device_input_shape_size, device_axes_shape_size, device_output_stride_size};
  if (data_type_ == kNumberTypeInt8 || data_type_ == kNumberTypeInt16 || data_type_ == kNumberTypeUInt8 ||
      data_type_ == kNumberTypeUInt16 || data_type_ == kNumberTypeFloat16) {
    const size_t device_middle_output = output_elements_ * sizeof(float);
    workspace_size_list_.emplace_back(device_middle_output);
  }
}

bool EuclideanNormGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int EuclideanNormGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  keep_dims_ = GetValue<bool>(primitive_->GetAttr(ops::kKeepDims));
  axes_ = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  data_type_ = inputs.at(kIndex0)->dtype_id();
  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());

  output_shape_.clear();
  if (axes_.size() == input_shape.size()) {
    output_shape_ = {1};
    output_elements_ = 1;
    InitWorkSpaceSizeList();
    return KRET_OK;
  }
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);
  output_axes_.clear();
  std::vector<size_t> axes;
  int64_t input_rank = SizeToLong(input_shape_.size());
  (void)std::transform(axes_.begin(), axes_.end(), std::back_inserter(axes), [&input_rank](const int64_t &dim) {
    return dim < 0 ? LongToSize(dim + input_rank) : LongToSize(dim);
  });
  std::set<size_t> axes_set(axes.begin(), axes.end());
  std::vector<size_t> output_shape_no_keep_dim;
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    if (!axes_set.count(i)) {
      output_axes_.emplace_back(i);
      if (keep_dims_) {
        output_shape_no_keep_dim.emplace_back(input_shape_[i]);
      }
    }
  }
  output_stride_.clear();
  output_stride_.resize(output_axes_.size());
  output_stride_[output_stride_.size() - 1] = 1;
  for (int i = static_cast<int>(output_stride_.size() - 2); i >= 0; --i) {
    output_stride_[i] = keep_dims_ ? output_stride_[i + 1] * output_shape_no_keep_dim[i + 1]
                                   : output_stride_[i + 1] * output_shape_[i + 1];
  }
  output_elements_ = std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>());
  InitWorkSpaceSizeList();
  return KRET_OK;
}

template <typename T>
bool EuclideanNormGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  size_t *device_input_shape = nullptr;
  size_t *device_axes_output = nullptr;
  size_t *device_output_stride = nullptr;
  if (workspace[kIndex0]->size() != 0) {
    device_input_shape = GetDeviceAddress<size_t>(workspace, kIndex0);
  }
  if (workspace[kIndex1]->size() != 0) {
    device_axes_output = GetDeviceAddress<size_t>(workspace, kIndex1);
  }
  if (workspace[kIndex2]->size() != 0) {
    device_output_stride = GetDeviceAddress<size_t>(workspace, kIndex2);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_input_shape, input_shape_.data(), input_shape_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "EuclideanNormGpuKernelMod cudaMemcpyAsync input_shape_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_axes_output, output_axes_.data(), output_axes_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "EuclideanNormGpuKernelMod cudaMemcpyAsync output_axes_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_output_stride, output_stride_.data(), output_stride_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "EuclideanNormGpuKernelMod cudaMemcpyAsync output_shape_ failed");

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(output, 0, output_elements_ * sizeof(T)),
                                    "EuclideanNormGpuKernelMod failed to set output cuda memory to zeros.");
  cudaError_t status = cudaErrorNotReady;
  if constexpr ((std::is_same_v<T, int8_t>) || (std::is_same_v<T, int16_t>) || (std::is_same_v<T, uint8_t>) ||
                (std::is_same_v<T, uint16_t>) || (std::is_same_v<T, half>)) {
    auto middle_output = GetDeviceAddress<float>(workspace, kIndex3);
    auto middle_output_size = output_elements_ * sizeof(float);
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(middle_output, 0, middle_output_size),
                                      "LpNormGpuKernelMod failed  to set middle output cuda memory to zeros.");
    status = CalEuclideanNorm(input, device_input_shape, input_shape_.size(), input_elements_, device_axes_output,
                              device_output_stride, output_axes_.size(), output_elements_, middle_output, output,
                              device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    status = CalEuclideanNorm(input, device_input_shape, input_shape_.size(), input_elements_, device_axes_output,
                              device_output_stride, output_axes_.size(), output_elements_, nullptr, output, device_id_,
                              reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
// clang-format off
std::vector<std::pair<KernelAttr, EuclideanNormGpuKernelMod::EuclideanNormFunc>> EuclideanNormGpuKernelMod::func_list_ =
  { // NOLINT
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &EuclideanNormGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &EuclideanNormGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &EuclideanNormGpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &EuclideanNormGpuKernelMod::LaunchKernel<int64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &EuclideanNormGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &EuclideanNormGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &EuclideanNormGpuKernelMod::LaunchKernel<double>},

    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &EuclideanNormGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &EuclideanNormGpuKernelMod::LaunchKernel<Complex<double>>},

    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &EuclideanNormGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &EuclideanNormGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &EuclideanNormGpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &EuclideanNormGpuKernelMod::LaunchKernel<int64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &EuclideanNormGpuKernelMod::LaunchKernel<uint64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &EuclideanNormGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &EuclideanNormGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &EuclideanNormGpuKernelMod::LaunchKernel<double>},

    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &EuclideanNormGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &EuclideanNormGpuKernelMod::LaunchKernel<Complex<double>>},
};
// clang-format on

std::vector<KernelAttr> EuclideanNormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EuclideanNormFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, EuclideanNorm, EuclideanNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
