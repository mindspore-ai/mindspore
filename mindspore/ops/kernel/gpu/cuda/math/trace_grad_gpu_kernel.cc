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

#include "kernel/gpu/cuda/math/trace_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t kColindex = 1;
constexpr size_t kRowindex = 2;
bool TraceGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [unit8, uint16, uint32, uint64, int8, "
                  << "int16, int32, int64, float16, float32, float64, bool], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;

  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  shape_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  return true;
}

int TraceGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool TraceGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *y_grad = GetDeviceAddress<T>(inputs, 0);
  S *input_shape = GetDeviceAddress<S>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  size_t input_shape_num = 2;
  S *input_shape_ptr = reinterpret_cast<S *>(malloc(input_shape_num * shape_unit_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape_ptr, input_shape, input_shape_num * shape_unit_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync input_shape failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - TraceGrad");
  S threads_num = input_shape_ptr[0] * input_shape_ptr[1];
  auto status =
    CalTraceGrad(threads_num, y_grad, input_shape, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, TraceGradGpuKernelMod::TraceGradFunc>> TraceGradGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &TraceGradGpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &TraceGradGpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &TraceGradGpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &TraceGradGpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &TraceGradGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &TraceGradGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &TraceGradGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &TraceGradGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &TraceGradGpuKernelMod::LaunchKernel<half, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &TraceGradGpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &TraceGradGpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &TraceGradGpuKernelMod::LaunchKernel<Complex<float>, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &TraceGradGpuKernelMod::LaunchKernel<Complex<double>, int32_t>},

  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &TraceGradGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &TraceGradGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &TraceGradGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &TraceGradGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &TraceGradGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &TraceGradGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &TraceGradGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &TraceGradGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &TraceGradGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &TraceGradGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &TraceGradGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &TraceGradGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &TraceGradGpuKernelMod::LaunchKernel<Complex<double>, int64_t>}};

std::vector<KernelAttr> TraceGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TraceGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TraceGrad, TraceGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
