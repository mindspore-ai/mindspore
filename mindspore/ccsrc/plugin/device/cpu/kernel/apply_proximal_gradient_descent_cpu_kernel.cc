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
#include "plugin/device/cpu/kernel/apply_proximal_gradient_descent_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32_grad/apply_proximal_gradient_descent_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/intrinsics/ms_simd_instructions.h"

namespace {
constexpr size_t kApplyProximalGradientDescentInputsNum = 5;
constexpr size_t kApplyProximalGradientDescentOutputsNum = 1;
constexpr size_t kVarIndex = 0;
constexpr size_t kAlphaIndex = 1;
constexpr size_t kL1Index = 2;
constexpr size_t kL2Index = 3;
constexpr size_t kDeltaIndex = 4;
template <typename T>
int Sgn(T val) {
  if (val > T(0)) {
    return 1;
  }
  if (val < T(0)) {
    return -1;
  }
  return 0;
}
template <typename T>
T Abs(T x) {
  if (x >= T(0)) {
    return x;
  }
  return -x;
}
template <typename T>
T Max(T x, T y) {
  if (x > y) {
    return x;
  }
  return y;
}
}  // namespace

namespace mindspore {
namespace kernel {
bool ApplyProximalGradientDescentCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();
  return true;
}

int ApplyProximalGradientDescentCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs,
                                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kApplyProximalGradientDescentInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 5.";
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> alpha_shape = inputs[kAlphaIndex]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kL1Index]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kL2Index]->GetShapeVector();
  std::vector<int64_t> delta_shape = inputs[kDeltaIndex]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, delta_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'delta' must be the same as the shape of 'var', "
                     "but got the shape of 'delta': "
                  << Vector2Str(delta_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(alpha_shape, l1_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'alpha' must be the same as the shape of 'l1', "
                     "but got the shape of 'alpha': "
                  << Vector2Str(alpha_shape) << " and the shape of 'l1': " << Vector2Str(l1_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(alpha_shape, l2_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'alpha' must be the same as the shape of 'l2', "
                     "but got the shape of 'alpha': "
                  << Vector2Str(alpha_shape) << " and the shape of 'l2': " << Vector2Str(l2_shape);
    return KRET_RESIZE_FAILED;
  }
  if (batch_rank_ < 0 || alpha_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'alpha' must be equal to 'batch_rank', "
                     "but got the shape of 'alpha': "
                  << Vector2Str(alpha_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  batch_size_ = 1;
  if (!alpha_shape.empty()) {
    batch_size_ = std::accumulate(alpha_shape.begin(), alpha_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = input_elements_ / LongToSize(batch_size_);
  if (batch_rank_ > 1) {
    if (var_shape.size() < alpha_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than 'alpha_shape', but got the shape of 'var': "
                    << Vector2Str(var_shape) << " and 'alpha_shape': " << Vector2Str(alpha_shape);
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(alpha_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'alpha', "
                       "but got the batch shape of 'var': "
                    << Vector2Str(var_batch_shape) << " and the shape of 'alpha': " << Vector2Str(alpha_shape);
      return KRET_RESIZE_FAILED;
    }
  }

  return ret;
}

template <typename T>
void ApplyProximalGradientDescentCpuKernelMod::LaunchKernelDefault(const std::vector<AddressPtr> &inputs,
                                                                   const std::vector<AddressPtr> &) {
  auto var_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto alpha_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto l1_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto l2_addr = reinterpret_cast<T *>(inputs[3]->addr);
  auto delta_addr = reinterpret_cast<T *>(inputs[4]->addr);
  auto task = [this, &var_addr, &alpha_addr, &l1_addr, &l2_addr, &delta_addr](size_t start, size_t end) {
    auto cur_input_elements = end - start;
    for (size_t b = 0; b < static_cast<size_t>(batch_size_); b++) {
      auto offset = b * input_elements_ + start;
      auto var_cur = var_addr + offset;
      auto delta_cur = delta_addr + offset;

      for (size_t pos = 0; pos < cur_input_elements; pos++) {
        T prox_var = var_cur[pos] - alpha_addr[b] * delta_cur[pos];
        if (l1_addr[b] > static_cast<T>(0)) {
          var_cur[pos] = static_cast<T>(Sgn(prox_var)) *
                         Max(Abs(prox_var) - alpha_addr[b] * l1_addr[b], static_cast<T>(0)) /
                         (static_cast<T>(1) + alpha_addr[b] * l2_addr[b]);
        } else {
          var_cur[pos] = prox_var / (static_cast<T>(1) + alpha_addr[b] * l2_addr[b]);
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);
}

void ApplyProximalGradientDescentCpuKernelMod::LaunchKernelOptFp32(const std::vector<kernel::AddressPtr> &inputs,
                                                                   const std::vector<kernel::AddressPtr> &) {
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->addr);
  auto alpha = reinterpret_cast<float *>(inputs[kAlphaIndex]->addr);
  auto l1 = reinterpret_cast<float *>(inputs[kL1Index]->addr);
  auto l2 = reinterpret_cast<float *>(inputs[kL2Index]->addr);
  auto delta = reinterpret_cast<float *>(inputs[kDeltaIndex]->addr);

  auto task = [this, &var, &alpha, &l1, &l2, &delta](size_t start, size_t end) {
    auto cur_input_elements = end - start;
    for (size_t b = 0; b < static_cast<size_t>(batch_size_); b++) {
      auto offset = b * input_elements_ + start;
      auto var_cur = var + offset;
      auto delta_cur = delta + offset;

      ApplyProximalGradientDescentOpt(var_cur, alpha[b], l1[b], l2[b], delta_cur, cur_input_elements);
    }
  };

  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);
}

bool ApplyProximalGradientDescentCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyProximalGradientDescentInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyProximalGradientDescentOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernelDefault<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernelOptFp32(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', input dtype only support float16 and float32, but got ["
                            << dtype_ << "].";
  }
  return true;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyProximalGradientDescent, ApplyProximalGradientDescentCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
