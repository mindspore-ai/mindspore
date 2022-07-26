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

#include "plugin/device/cpu/kernel/apply_ftrl_cpu_kernel.h"
#include <map>
#include <functional>
#include <algorithm>
#include "utils/ms_utils.h"
#include "mindspore/core/ops/apply_ftrl.h"
#include "ops/apply_ftrl.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyFtrlInputsNum = 8;
constexpr size_t kApplyFtrlOutputsNum = 1;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexAcc = 1;
constexpr size_t kIndexLinear = 2;
constexpr size_t kIndexGrad = 3;
constexpr size_t kIndexLR = 4;
constexpr size_t kIndexL1 = 5;
constexpr size_t kIndexL2 = 6;
constexpr size_t kIndexLRPower = 7;
constexpr size_t kIndexOutput = 0;
}  // namespace

bool ApplyFtrlCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();

  if (inputs.size() != kApplyFtrlInputsNum || outputs.size() != kApplyFtrlOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's inputs and output size should be " << kApplyFtrlInputsNum
                  << " and " << kApplyFtrlOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  return true;
}

int ApplyFtrlCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, others)) != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> var_shape = inputs[kIndexVar]->GetShapeVector();
  std::vector<int64_t> acc_shape = inputs[kIndexAcc]->GetShapeVector();
  std::vector<int64_t> linear_shape = inputs[kIndexLinear]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndexGrad]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLR]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kIndexL1]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kIndexL2]->GetShapeVector();
  std::vector<int64_t> lr_power_shape = inputs[kIndexLRPower]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, acc_shape) || !IsSameShape(var_shape, linear_shape) ||
      !IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'linear', 'acc', 'grad' and 'var' must be the same, "
                  << "but get the shapes of 'acc': " << Vector2Str(acc_shape)
                  << ", 'linear': " << Vector2Str(linear_shape) << ", 'grad': " << Vector2Str(grad_shape)
                  << " and 'var': " << Vector2Str(var_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l1_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l1', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and the shape of 'l1': " << Vector2Str(l1_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l2_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l2', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and the shape of 'l2': " << Vector2Str(l2_shape);
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, lr_power_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'lr_power_shape', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and the shape of 'lr_power_shape': " << Vector2Str(lr_power_shape);
    return KRET_RESIZE_FAILED;
  }

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), 1, std::multiplies<int64_t>());
  }

  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  int64_t temp_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  input_elements_ = static_cast<size_t>(temp_elements_ / batch_size_);

  return 0;
}

template <typename T>
void ApplyFtrlCpuKernelMod::LaunchApplyFtrl(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *accum = reinterpret_cast<T *>(inputs[kIndexAcc]->addr);
  T *linear = reinterpret_cast<T *>(inputs[kIndexLinear]->addr);
  T *grad = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);
  T *lr = reinterpret_cast<T *>(inputs[kIndexLR]->addr);
  T *l1 = reinterpret_cast<T *>(inputs[kIndexL1]->addr);
  T *l2 = reinterpret_cast<T *>(inputs[kIndexL2]->addr);
  T *lr_power = reinterpret_cast<T *>(inputs[kIndexLRPower]->addr);

  for (int64_t b = 0; b < batch_size_; b++) {
    auto task = [this, &var, &accum, &linear, &grad, &lr, &l1, &l2, &lr_power](size_t start, size_t end) {
      auto two = static_cast<T>(2.0);
      const T learning_rate_power_val = -lr_power[0];
      for (size_t i = start; i < end; ++i) {
        auto cur_accum = accum[i] + grad[i] * grad[i];
        const T accum_power = pow(accum[i], learning_rate_power_val);
        const T cur_accum_power = pow(cur_accum, learning_rate_power_val);
        linear[i] += grad[i] - (cur_accum_power - accum_power) / lr[0] * var[i];
        if (abs(linear[i]) > l1[0]) {
          auto sign_linear_mul_l1 = linear[i] ? l1[0] : -l1[0];
          auto denominator = cur_accum_power / lr[0] + two * l2[0];
          var[i] = (sign_linear_mul_l1 - linear[i]) / denominator;
        } else {
          var[i] = static_cast<T>(0);
        }
        accum[i] = cur_accum;
      }
    };
    ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_);
    var = var + input_elements_;
    accum = accum + input_elements_;
    linear = linear + input_elements_;
    grad = grad + input_elements_;
  }
}

bool ApplyFtrlCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs[kIndexVar]->size != inputs[kIndexAcc]->size) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape and dtype of 'acc' and 'var' should be same, but got the memory size of 'acc': "
                  << inputs[kIndexAcc]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexLinear]->size) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', the shape and dtype of 'linear' and 'var' should be same, but got the memory size of 'linear': "
      << inputs[kIndexLinear]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexGrad]->size) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape and dtype of 'grad' and 'var' should be same, but got the memory size of 'grad': "
                  << inputs[kIndexGrad]->size << " and 'var': " << inputs[kIndexVar]->size;
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchApplyFtrl<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchApplyFtrl<float16>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of 'var' should be float16 or float32, but get "
                  << TypeIdToType(dtype_)->ToString();
  }

  return true;
}

std::vector<KernelAttr> ApplyFtrlCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutInRef(0, 0),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutInRef(0, 0)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyFtrl, ApplyFtrlCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
