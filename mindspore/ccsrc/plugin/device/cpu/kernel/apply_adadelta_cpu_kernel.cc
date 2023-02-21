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

#include <algorithm>

#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/apply_adadelta_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/intrinsics/ms_simd_instructions.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
constexpr size_t kApplyAdadeltaInputsNum = 7;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kAccumUpdateIndex = 2;
constexpr size_t kLRIndex = 3;
constexpr size_t kRhoIndex = 4;
constexpr size_t kEpsilonIndex = 5;
constexpr size_t kGradIndex = 6;

bool ApplyAdadeltaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();

  auto input_type_id = inputs[0]->GetDtype();
  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);

  return true;
}

int ApplyAdadeltaCpuKernelMod::CheckInputShape(const std::vector<KernelTensorPtr> &inputs) {
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kAccumIndex]->GetShapeVector();
  std::vector<int64_t> accum_update_shape = inputs[kAccumUpdateIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();
  std::vector<int64_t> rho_shape = inputs[kRhoIndex]->GetShapeVector();
  std::vector<int64_t> epsilon_shape = inputs[kEpsilonIndex]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  if (!(IsSameShape(var_shape, accum_shape) && IsSameShape(var_shape, accum_update_shape) &&
        IsSameShape(var_shape, grad_shape))) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'var', 'accum', 'accum_update', 'grad' "
                     "must be the same, "
                     "but got the shapes 'var': "
                  << Vector2Str(var_shape) << ", 'accum': " << Vector2Str(accum_shape)
                  << ", 'accum_update': " << Vector2Str(accum_update_shape) << ", 'grad': " << Vector2Str(grad_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!(IsSameShape(lr_shape, rho_shape) || IsSameShape(lr_shape, epsilon_shape))) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr', 'rho' and 'epsilon' must be the same, "
                     "but got the shapes 'lr': "
                  << Vector2Str(lr_shape) << ", 'rho': " << Vector2Str(rho_shape)
                  << ", 'epsilon': " << Vector2Str(epsilon_shape);

    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int ApplyAdadeltaCpuKernelMod::CheckShapeSize(std::vector<int64_t> var_shape, std::vector<int64_t> lr_shape) {
  if (batch_rank_ > 1) {
    if (var_shape.size() < lr_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than "
                       "'lr_shape', but got the shape of 'var': "
                    << Vector2Str(var_shape) << " and 'lr_shape': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the "
                       "shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << Vector2Str(var_batch_shape) << " and the shape of 'lr': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

int ApplyAdadeltaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  if (input_size_list_.size() != kApplyAdadeltaInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 7.";
    return KRET_RESIZE_FAILED;
  }
  ret = CheckInputShape(inputs);
  if (ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();

  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;

  ret = CheckShapeSize(var_shape, lr_shape);

  return ret;
}

bool ApplyAdadeltaCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdadeltaInputsNum, kernel_name_);
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<float *>(inputs[kAccumIndex]->addr);
  auto accum_update = reinterpret_cast<float *>(inputs[kAccumUpdateIndex]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr);
  auto rho = reinterpret_cast<float *>(inputs[kRhoIndex]->addr);
  auto epsilon = reinterpret_cast<float *>(inputs[kEpsilonIndex]->addr);
  auto grad = reinterpret_cast<float *>(inputs[kGradIndex]->addr);

  for (int64_t b = 0; b < batch_size_; b++) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        accum[i] = rho[b] * accum[i] + (1.0 - rho[b]) * (grad[i] * grad[i]);
        float update = sqrt(accum_update[i] + epsilon[b]) * (grad[i] / sqrt(accum[i] + epsilon[b]));
        accum_update[i] = rho[b] * accum_update[i] + (1.0 - rho[b]) * (update * update);
        var[i] -= lr[b] * update;
      }
    };
    ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_);

    var = var + input_elements_;
    accum = accum + input_elements_;
    accum_update = accum_update + input_elements_;
    grad = grad + input_elements_;
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdadelta, ApplyAdadeltaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
