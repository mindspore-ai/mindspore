/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/random/parameterized_truncated_normal_gpu_kernel.h"
#include "mindspore/core/ops/parameterized_truncated_normal.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/parameterized_truncated_normal_impl.cuh"
#include <utility>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "include/curand_kernel.h"

namespace mindspore {
namespace kernel {
using KernelRunFunc = ParameterizedTruncatedNormalGpuKernelMod::KernelRunFunc;

bool ParameterizedTruncatedNormalGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  // the size of per element of args
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);

  // setup seed
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed2")));
  seed_ = random::GetSeed(seed, seed2);

  return true;
}

int ParameterizedTruncatedNormalGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs,
                                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  if (!IsValidShape(output_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  ResetResource();

  std::vector<int64_t> mean_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                         inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> stdevs_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> min_shape = std::vector<int64_t>(inputs.at(kIndex3)->GetDeviceShapeAdaptively().begin(),
                                                        inputs.at(kIndex3)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> max_shape = std::vector<int64_t>(inputs.at(kIndex4)->GetDeviceShapeAdaptively().begin(),
                                                        inputs.at(kIndex4)->GetDeviceShapeAdaptively().end());
  int64_t mean_elements = std::accumulate(mean_shape.begin(), mean_shape.end(), 1, std::multiplies<int64_t>());
  stdevs_elements_ = std::accumulate(stdevs_shape.begin(), stdevs_shape.end(), 1, std::multiplies<int64_t>());
  int64_t min_elements = std::accumulate(min_shape.begin(), min_shape.end(), 1, std::multiplies<int64_t>());
  int64_t max_elements = std::accumulate(max_shape.begin(), max_shape.end(), 1, std::multiplies<int64_t>());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());

  scalar_mean_ = (mean_elements == 1);
  scalar_stdevs_ = (stdevs_elements_ == 1);
  scalar_min_ = (min_elements == 1);
  scalar_max_ = (max_elements == 1);

  batch_size_ = output_shape[0];
  samples_per_batch_ = output_elements_ / batch_size_;

  if (output_elements_ == 0) {
    is_null_input_ = true;
  }

  input_size_list_.emplace_back(mean_elements * unit_output_size_);
  input_size_list_.emplace_back(stdevs_elements_ * unit_output_size_);
  input_size_list_.emplace_back(min_elements * unit_output_size_);
  input_size_list_.emplace_back(max_elements * unit_output_size_);
  output_size_list_.emplace_back(output_elements_ * unit_output_size_);

  return KRET_OK;
}

template <typename T>
bool ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<AddressPtr> &workspace,
                                                            const std::vector<kernel::AddressPtr> &outputs) {
  T *mean = GetDeviceAddress<T>(inputs, kIndex1);
  T *stdevs = GetDeviceAddress<T>(inputs, kIndex2);
  T *min = GetDeviceAddress<T>(inputs, kIndex3);
  T *max = GetDeviceAddress<T>(inputs, kIndex4);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  // check stdevs value
  T zero = 0.;
  std::vector<T> stdevs_host;
  stdevs_host.resize(stdevs_elements_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(stdevs_host.data(), stdevs, stdevs_elements_ * sizeof(T), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'PrarmeterizedTruncatedNormal', cudaMemcpy for 'stdevs' failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'PrarmeterizedTruncatedNormal', cudaStreamSyncFailed");
  }
  for (int64_t i = 0; i < stdevs_elements_; i++) {
    if (stdevs_host[i] <= zero) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'stdevs' should be greater than zero.";
    }
  }

  // launch kernel function
  auto status = ParameterizedTruncatedNormal(seed_, seed_offset_, batch_size_, samples_per_batch_, mean, stdevs, min,
                                             max, output, scalar_mean_, scalar_stdevs_, scalar_min_, scalar_max_,
                                             device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  seed_offset_ += 1;
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ParameterizedTruncatedNormalGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ParameterizedTruncatedNormalGpuKernelMod::LaunchKernel<double>},
  };

  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ParameterizedTruncatedNormal, ParameterizedTruncatedNormalGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
