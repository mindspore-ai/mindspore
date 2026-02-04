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

#include "kernel/gpu/cuda/arrays/conjugate_transpose_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <memory>
#include <complex>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "mindspore/ops/infer/conjugate_transpose.h"
#include "kernel/gpu/cuda_impl/cuda_ops/conjugate_transpose_impl.cuh"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
template <typename U>
using Complex = mindspore::utils::Complex<U>;
constexpr int MAX_DIMS = 7;
bool ConjugateTransposeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_one_ = abstract::TypeIdSize(inputs[0]->dtype_id());
  unit_size_two_ = abstract::TypeIdSize(inputs[1]->dtype_id());
  out_unit_size_ = abstract::TypeIdSize(outputs[0]->dtype_id());

  return true;
}

int ConjugateTransposeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto x_long_shape = inputs.at(kIndex0)->GetShapeVector();
  auto x_two_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<size_t> x_shape_one;
  std::vector<size_t> x_shape_two;
  (void)std::transform(x_long_shape.begin(), x_long_shape.end(), std::back_inserter(x_shape_one), LongToSize);
  if (x_shape_one.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got x: " << x_shape_one.size();
  }
  (void)std::transform(x_two_shape.begin(), x_two_shape.end(), std::back_inserter(x_shape_two), LongToSize);
  auto y_long_shape = outputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> y_shape;
  (void)std::transform(y_long_shape.begin(), y_long_shape.end(), std::back_inserter(y_shape), LongToSize);
  if (y_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got y: " << y_shape.size();
  }
  shape_size_ = x_shape_one.size();
  x_one_shape_.resize(MAX_DIMS, 1);
  x_two_shape_.resize(MAX_DIMS, 1);
  y_shape_.resize(MAX_DIMS, 1);
  x_one_count_ = (shape_size_ > 0) ? x_one_count_ * x_shape_one[0] : x_one_count_;
  input_stride[shape_size_ - 1] = 1;
  for (size_t i = 1; i < x_shape_one.size(); i++) {
    x_one_count_ *= x_shape_one[i];
    input_stride[shape_size_ - 1 - i] = input_stride[shape_size_ - i] * x_shape_one[shape_size_ - i];
  }
  x_two_count_ *= x_shape_two[0];
  for (size_t i = 1; i < x_shape_two.size(); i++) {
    x_two_count_ *= x_shape_two[i];
  }
  y_count_ = (y_shape.size() > 0) ? y_count_ * y_shape[0] : y_count_;
  output_stride[shape_size_ - 1] = 1;
  for (size_t i = 1; i < y_shape.size(); i++) {
    y_count_ *= y_shape[i];
    output_stride[shape_size_ - 1 - i] = output_stride[shape_size_ - i] * y_shape[shape_size_ - i];
  }
  size_t workspace_size_ = MAX_DIMS * sizeof(size_t);
  size_t output_size = y_count_ * out_unit_size_;
  output_size_list_.push_back(output_size);
  workspace_size_list_.push_back(workspace_size_);
  workspace_size_list_.push_back(workspace_size_);
  return KRET_OK;
}

void ConjugateTransposeGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  shape_size_ = 0;
  x_one_count_ = 1;
  x_two_count_ = 1;
  y_count_ = 1;
  x_one_shape_.clear();
  x_two_shape_.clear();
  y_shape_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T, typename S>
bool ConjugateTransposeGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  T *y = GetDeviceAddress<T>(outputs, kIndex0);
  S *input_axis = GetDeviceAddress<S>(inputs, kIndex1);
  size_t *input_stride_ = GetDeviceAddress<size_t>(workspace, kIndex0);
  size_t *output_stride_ = GetDeviceAddress<size_t>(workspace, kIndex1);

  size_t size = x_one_count_;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_stride_, &input_stride[0], sizeof(size_t) * MAX_DIMS, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For " << "input_stride_ "
           << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_stride_, &output_stride[0], sizeof(size_t) * MAX_DIMS, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For " << "output_stride_ "
           << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(y, 0, outputs[kIndex0]->size(), reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "ConjugateTransposeGpuKernelMod cudaMemSet Failed");
  auto status = CalConjugateTranspose(size, x, input_stride_, output_stride_, input_axis, shape_size_, y, device_id_,
                                      reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T, typename S>
bool ConjugateTransposeGpuKernelMod::LaunchComplexKernel(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &workspace,
                                                         const std::vector<KernelTensor *> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  T *y = GetDeviceAddress<T>(outputs, kIndex0);
  S *input_axis = GetDeviceAddress<S>(inputs, kIndex1);
  size_t *input_stride_ = GetDeviceAddress<size_t>(workspace, kIndex0);
  size_t *output_stride_ = GetDeviceAddress<size_t>(workspace, kIndex1);

  size_t size = x_one_count_;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_stride_, &input_stride[0], sizeof(size_t) * MAX_DIMS, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For " << "input_stride_ "
           << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_stride_, &output_stride[0], sizeof(size_t) * MAX_DIMS, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For " << "output_stride_ "
           << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(y, 0, outputs[kIndex0]->size(), reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "ConjugateTransposeGpuKernelMod cudaMemSet Failed");
  auto status = CalConjugateTransposeComplex(size, x, input_stride_, output_stride_, input_axis, shape_size_, y,
                                             device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ConjugateTransposeGpuKernelMod::ConjugateTransposeFunc>>
  ConjugateTransposeGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &ConjugateTransposeGpuKernelMod::LaunchComplexKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &ConjugateTransposeGpuKernelMod::LaunchComplexKernel<Complex<double>, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<int, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &ConjugateTransposeGpuKernelMod::LaunchKernel<uint32_t, int64_t>}};

std::vector<KernelAttr> ConjugateTransposeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ConjugateTransposeFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ConjugateTranspose, ConjugateTransposeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
