/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/softplus_ext_cpu_kernel.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include "kernel/cpu/nnacl/errorcode.h"
#include "kernel/cpu/nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace kernel {
namespace softplus_ext_cpu {
namespace {
#define SOFTPLUS_EXT_CPU_REGISTER(TENSOR_DTYPE, BETA_DTYPE, THRESHOLD_DTYPE, T, TBeta, TThreshold) \
  KernelAttr()                                                                                     \
    .AddInputAttr(TENSOR_DTYPE)                                                                    \
    .AddInputAttr(kObjectTypeNumber, BETA_DTYPE)                                                   \
    .AddInputAttr(kObjectTypeNumber, THRESHOLD_DTYPE)                                              \
    .AddOutputAttr(TENSOR_DTYPE),                                                                  \
    &SoftplusExtCpuKernelMod::LaunchKernel<T, TBeta, TThreshold>
}  // namespace

bool SoftplusExtCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', got empty inputs or outputs, which is invalid.";
    return false;
  }
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SoftplusExtCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t{1}, std::multiplies<size_t>());
  return KRET_OK;
}

template <typename T, typename TBeta, typename TThreshold>
bool SoftplusExtCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &outputs) {
  using ComputeT = std::conditional_t<std::is_same_v<T, double>, double, float>;
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  const ComputeT beta = static_cast<ComputeT>(inputs[kIndex1]->GetValueWithCheck<TBeta>());
  const ComputeT threshold = static_cast<ComputeT>(inputs[kIndex2]->GetValueWithCheck<TThreshold>());

  if (beta == static_cast<ComputeT>(0)) {
    MS_EXCEPTION(ValueError) << "For 'SoftplusExt', 'beta' must not be 0.";
  }

  if constexpr (std::is_same_v<T, float>) {
    if (beta == static_cast<ComputeT>(1) && threshold == static_cast<ComputeT>(20)) {
      auto task = [input, output, this](size_t start, size_t end) {
        auto ret = ::Softplus(input + start, SizeToInt(end - start), output + start);
        if (ret != NNACL_OK) {
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', call NNACL Softplus function failed. Error code: " << ret;
          return false;
        }
        return true;
      };
      ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);
      return true;
    }
  }

  auto task = [input, output, beta, threshold](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const ComputeT x = static_cast<ComputeT>(input[i]);
      const ComputeT scaled = x * beta;
      if (scaled > threshold) {
        output[i] = static_cast<T>(x);
        continue;
      }
      const ComputeT softplus = scaled > static_cast<ComputeT>(0) ? (scaled + std::log1p(std::exp(-scaled))) / beta
                                                                  : std::log1p(std::exp(scaled)) / beta;
      output[i] = static_cast<T>(softplus);
    }
  };
  ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, SoftplusExtCpuKernelMod::KernelRunFunc>> &SoftplusExtCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SoftplusExtCpuKernelMod::KernelRunFunc>> func_list = {
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, float, float, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat64, float, float, double)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeInt64, float, float, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat32, float, double, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat64, float, double, double)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64, float, int64_t, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float, int64_t, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeInt64, float, double, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat64, float, int64_t, double)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeFloat32, double, float, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeFloat64, double, float, double)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeInt64, double, float, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat32, double, double, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64, double, double, double)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt64, double, int64_t, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat32, double, int64_t, float)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeInt64, double, double, int64_t)},
    {SOFTPLUS_EXT_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double, int64_t, double)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SoftplusExt, SoftplusExtCpuKernelMod);
}  // namespace softplus_ext_cpu
}  // namespace kernel
}  // namespace mindspore
