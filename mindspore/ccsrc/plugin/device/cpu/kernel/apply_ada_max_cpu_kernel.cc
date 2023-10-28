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

#include <cmath>
#include <map>
#include <functional>
#include "plugin/device/cpu/kernel/apply_ada_max_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace {
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;

constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexBeta1Power = 3;
constexpr size_t kIndexLr = 4;
constexpr size_t kIndexBeta1 = 5;
constexpr size_t kIndexBeta2 = 6;
constexpr size_t kIndexEpsilon = 7;
constexpr size_t kIndexGrad = 8;

constexpr size_t kApplyAdaMaxInputsNum = 9;
constexpr size_t kApplyAdaMaxOutputsNum = 3;

bool CheckShapeIsScalar(const ShapeVector &shape) { return shape.empty() || (shape.size() == 1 && shape[0] == 1); }
}  // namespace

namespace mindspore {
namespace kernel {
bool ApplyAdaMaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();
  return true;
}

int ApplyAdaMaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  std::vector<int64_t> var_shape = inputs[kIndexVar]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kIndexM]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kIndexV]->GetShapeVector();
  std::vector<int64_t> beta1_power_shape = inputs[kIndexBeta1Power]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLr]->GetShapeVector();
  std::vector<int64_t> beta1_shape = inputs[kIndexBeta1]->GetShapeVector();
  std::vector<int64_t> beta2_shape = inputs[kIndexBeta2]->GetShapeVector();
  std::vector<int64_t> epsilon_shape = inputs[kIndexEpsilon]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndexGrad]->GetShapeVector();

  if (batch_rank_ == 0) {
    if (!CheckShapeIsScalar(beta1_power_shape) || !CheckShapeIsScalar(lr_shape) || !CheckShapeIsScalar(beta1_shape) ||
        !CheckShapeIsScalar(beta2_shape) || !CheckShapeIsScalar(epsilon_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'beta1_power'(shape: " << beta1_power_shape
                    << "), 'lr'(shape: " << lr_shape << "), 'beta1'(shape: " << beta1_shape
                    << "), 'beta2'(shape: " << beta2_shape << "), 'epsilon'(shape: " << epsilon_shape
                    << ") must be scalar";
      return KRET_RESIZE_FAILED;
    }
  } else {
    if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'lr' must be equal to 'batch_rank', but got the shape of 'lr': "
                    << lr_shape << " and 'batch_rank': " << batch_rank_;
      return KRET_RESIZE_FAILED;
    }
  }

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (m_shape != var_shape || v_shape != var_shape || grad_shape != var_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var': " << var_shape << ", 'm': " << m_shape
                  << ", 'v': " << v_shape << ", and 'grad': " << grad_shape << " must be equal.";
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), 1, std::multiplies<int64_t>());
  }

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  if (batch_size_ > 0) {
    input_elements_ = input_elements_ / batch_size_;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', batch size must be greater than 0, but got " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
  return ret;
}

bool ApplyAdaMaxCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdaMaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdaMaxOutputsNum, kernel_name_);
  if (inputs[kIndexVar]->size != inputs[kIndexM]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'm' and 'var' must be the same, but got the memory size of 'm': "
                      << inputs[kIndexM]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexV]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'v' and 'var' must be the same, but got the memory size of 'v': "
                      << inputs[kIndexV]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexGrad]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'grad' and 'var' must be the same, "
                         "but got the memory size of 'grad': "
                      << inputs[kIndexGrad]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', input dtype only support float16 and float32, but got ["
                            << dtype_ << "].";
  }
  return true;
}

template <typename T>
void ApplyAdaMaxCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndexV]->addr);
  T *beta1_power = reinterpret_cast<T *>(inputs[kIndexBeta1Power]->addr);
  T *lr = reinterpret_cast<T *>(inputs[kIndexLr]->addr);
  T *beta1 = reinterpret_cast<T *>(inputs[kIndexBeta1]->addr);
  T *beta2 = reinterpret_cast<T *>(inputs[kIndexBeta2]->addr);
  T *epsilon = reinterpret_cast<T *>(inputs[kIndexEpsilon]->addr);
  T *grad = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);

  auto one = static_cast<T>(1);
  if (beta1_power[kScalarIndex] == one) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' can't be 1.";
  }

  // multithreading
  for (int64_t b = 0; b < batch_size_; b++) {
    auto task = [this, &var, &m, &v, &beta1_power, &lr, &beta1, &beta2, &epsilon, &grad](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] = static_cast<T>(beta1[kScalarIndex] * m[i] + (one - beta1[kScalarIndex]) * grad[i]);
        auto zero = static_cast<T>(0);
        auto grad_abs = (grad[i] > zero) ? grad[i] : -grad[i];
        v[i] = std::max(beta2[kScalarIndex] * v[i], grad_abs);
        var[i] =
          var[i] - (lr[kScalarIndex] / (one - beta1_power[kScalarIndex])) * (m[i] / (v[i] + epsilon[kScalarIndex]));
      }
    };
    CPUKernelUtils::ParallelForAutoSearch(task, input_elements_, &parallel_search_info_);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    grad = grad + input_elements_;
    lr++;
    beta1++;
    beta1_power++;
    beta2++;
    epsilon++;
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdaMax, ApplyAdaMaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
