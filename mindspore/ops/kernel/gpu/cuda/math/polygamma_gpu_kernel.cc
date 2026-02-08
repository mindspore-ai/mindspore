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

#include "kernel/gpu/cuda/math/polygamma_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/digamma_impl.cuh"

namespace mindspore {
namespace kernel {
bool PolygammaGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  constexpr int INPUT_NUM = 2;
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << inputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  return true;
}

int PolygammaGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();

  std::vector<size_t> output_shape = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                         outputs.at(kIndex0)->GetDeviceShapeVector().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  size_t calc_shape = output_elements_ * input_size_;
  output_size_list_.push_back(calc_shape);
  return KRET_OK;
}

template <typename T1, typename T2>
bool PolygammaGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T1 *a = GetDeviceAddress<T1>(inputs, 0);
  T2 *input = GetDeviceAddress<T2>(inputs, 1);
  T2 *output = GetDeviceAddress<T2>(outputs, 0);
  T1 a_val;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&a_val, a, sizeof(T1), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For '" << kernel_name_ << "', "
            << "cudaMemcpy input 'a' to host failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For '" << kernel_name_ << "', "
                                               << "cudaStreamSyncFailed");
  }
  cudaError_t status = cudaErrorNotReady;
  if (a_val == static_cast<T1>(0)) {
    status = CalDigamma(output_elements_, input, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    status = CalPolygamma(output_elements_, a, input, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, PolygammaGpuKernelMod::PolygammaFunc>> PolygammaGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PolygammaGpuKernelMod::LaunchKernel<int32_t, half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PolygammaGpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &PolygammaGpuKernelMod::LaunchKernel<int32_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PolygammaGpuKernelMod::LaunchKernel<int64_t, half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PolygammaGpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &PolygammaGpuKernelMod::LaunchKernel<int64_t, double>}};

std::vector<KernelAttr> PolygammaGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PolygammaFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Polygamma, PolygammaGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
