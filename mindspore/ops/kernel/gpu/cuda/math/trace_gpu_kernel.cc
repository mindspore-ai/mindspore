/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/gpu/cuda/math/trace_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t kColindex = 1;
constexpr size_t kRowindex = 2;
bool TraceGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto prim = primitive_;
  MS_EXCEPTION_IF_NULL(prim);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [unit8, uint16, uint32, uint64, int8, "
                  << "int16, int32, int64, float16, float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}
int TraceGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeVector().end());
  size_t input_dims = input_shape.size();
  if (input_dims <= kIndex1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be at least 1-D, but got " << input_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  if (input_dims >= kIndex3) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be at most 2-D, but got " << input_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  matrix_row_ = input_shape[input_dims - kRowindex];
  matrix_col_ = input_shape[input_dims - kColindex];
  output_size_list_.push_back(unit_size_);
  return KRET_OK;
}

template <typename T>
bool TraceGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  size_t sum_size = std::min(matrix_col_, matrix_row_);
  auto status =
    CalTrace(input, sum_size, matrix_col_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, TraceGpuKernelMod::TraceFunc>> TraceGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &TraceGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &TraceGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &TraceGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &TraceGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), &TraceGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &TraceGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &TraceGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &TraceGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &TraceGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &TraceGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &TraceGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &TraceGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &TraceGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> TraceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TraceFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Trace, TraceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
