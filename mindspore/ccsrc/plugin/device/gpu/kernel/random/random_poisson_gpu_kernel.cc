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

#include "plugin/device/gpu/kernel/random/random_poisson_gpu_kernel.h"
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <algorithm>
#include "ir/anf.h"
#include "utils/log_adapter.h"
#include "kernel/common_utils.h"
#include "include/cuda_fp16.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = RandomPoissonGpuKernelMod::KernelRunFunc;
#define ADD_KERNEL(shape_dtype, rate_dtype, output_dtype, rate_type, output_type) \
  {                                                                               \
    KernelAttr()                                                                  \
      .AddInputAttr(kNumberType##shape_dtype)                                     \
      .AddInputAttr(kNumberType##rate_dtype)                                      \
      .AddOutputAttr(kNumberType##output_dtype),                                  \
      &RandomPoissonGpuKernelMod::LaunchKernel<rate_type, output_type>            \
  }
}  // namespace

bool RandomPoissonGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed2")));
  seed_ = random::GetSeed(seed, seed2);
  return true;
}

int RandomPoissonGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  std::vector<int64_t> rate_shape = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  rate_elements_ = std::accumulate(rate_shape.begin(), rate_shape.end(), 1, std::multiplies<int64_t>());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  workspace_size_list_.push_back(output_elements_ * sizeof(curandState));
  return KRET_OK;
}

template <typename R, typename T>
bool RandomPoissonGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  R *rate_addr = GetDeviceAddress<R>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
  MS_EXCEPTION_IF_NULL(rate_addr);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(workspace_addr);
  curandState *devStates = nullptr;
  devStates = reinterpret_cast<curandState *>(workspace_addr);
  auto status = RandomPoisson(seed_, seed_offset_, devStates, rate_addr, rate_elements_, output, output_elements_,
                              reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  seed_offset_ += 1;
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &RandomPoissonGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    ADD_KERNEL(Int32, Float16, Float16, half, half),     ADD_KERNEL(Int32, Float16, Float32, half, float),
    ADD_KERNEL(Int32, Float16, Float64, half, double),   ADD_KERNEL(Int32, Float16, Int32, half, int),
    ADD_KERNEL(Int32, Float16, Int64, half, int64_t),

    ADD_KERNEL(Int32, Float32, Float16, float, half),    ADD_KERNEL(Int32, Float32, Float32, float, float),
    ADD_KERNEL(Int32, Float32, Float64, float, double),  ADD_KERNEL(Int32, Float32, Int32, float, int),
    ADD_KERNEL(Int32, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int32, Float64, Float16, double, half),   ADD_KERNEL(Int32, Float64, Float32, double, float),
    ADD_KERNEL(Int32, Float64, Float64, double, double), ADD_KERNEL(Int32, Float64, Int32, double, int),
    ADD_KERNEL(Int32, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int32, Int32, Float16, int, half),        ADD_KERNEL(Int32, Int32, Float32, int, float),
    ADD_KERNEL(Int32, Int32, Float64, int, double),      ADD_KERNEL(Int32, Int32, Int32, int, int),
    ADD_KERNEL(Int32, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int32, Int64, Float16, int64_t, half),    ADD_KERNEL(Int32, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int32, Int64, Float64, int64_t, double),  ADD_KERNEL(Int32, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int32, Int64, Int64, int64_t, int64_t),

    ADD_KERNEL(Int64, Float16, Float16, half, half),     ADD_KERNEL(Int64, Float16, Float32, half, float),
    ADD_KERNEL(Int64, Float16, Float64, half, double),   ADD_KERNEL(Int64, Float16, Int32, half, int),
    ADD_KERNEL(Int64, Float16, Int64, half, int64_t),

    ADD_KERNEL(Int64, Float32, Float16, float, half),    ADD_KERNEL(Int64, Float32, Float32, float, float),
    ADD_KERNEL(Int64, Float32, Float64, float, double),  ADD_KERNEL(Int64, Float32, Int32, float, int),
    ADD_KERNEL(Int64, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int64, Float64, Float16, double, half),   ADD_KERNEL(Int64, Float64, Float32, double, float),
    ADD_KERNEL(Int64, Float64, Float64, double, double), ADD_KERNEL(Int64, Float64, Int32, double, int),
    ADD_KERNEL(Int64, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int64, Int32, Float16, int, half),        ADD_KERNEL(Int64, Int32, Float32, int, float),
    ADD_KERNEL(Int64, Int32, Float64, int, double),      ADD_KERNEL(Int64, Int32, Int32, int, int),
    ADD_KERNEL(Int64, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int64, Int64, Float16, int64_t, half),    ADD_KERNEL(Int64, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int64, Int64, Float64, int64_t, double),  ADD_KERNEL(Int64, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RandomPoisson, RandomPoissonGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
