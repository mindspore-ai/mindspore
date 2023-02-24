/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include "mindspore/core/ops/adam.h"
#include "plugin/device/cpu/kernel/adam_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdamInputsNum = 10;
constexpr size_t kAdamOutputsNum = 3;
constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexBeta1Power = 3;
constexpr size_t kIndexBeta2Power = 4;
constexpr size_t kIndexLr = 5;
constexpr size_t kIndexBeta1 = 6;
constexpr size_t kIndexBeta2 = 7;
constexpr size_t kIndexEpsilon = 8;
constexpr size_t kIndexGrad = 9;
constexpr float kAdamBlock = 1000;
}  // namespace

template <typename T>
void AdamCpuKernelMod::LaunchAdam(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndexV]->addr);
  float *beta1_power = reinterpret_cast<float *>(inputs[kIndexBeta1Power]->addr);
  float *beta2_power = reinterpret_cast<float *>(inputs[kIndexBeta2Power]->addr);
  float *lr = reinterpret_cast<float *>(inputs[kIndexLr]->addr);
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta1]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta2]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexEpsilon]->addr)[kScalarIndex]);
  T *gradient = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);
  constexpr float ONE = 1.0;

  for (int64_t b = 0; b < batch_size_; b++) {
    T new_lr = static_cast<T>(lr[b] * std::sqrt(ONE - beta2_power[b]) / (ONE - beta1_power[b]));
    // multithreading
    auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] += (gradient[i] - m[i]) * (one - beta1);
        v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2);
        T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
        if (use_nesterov_) {
          var[i] -= new_lr * (m[i] * beta1 + (one - beta1) * gradient[i]) / (sqrt_v + epsilon);
        } else {
          var[i] -= new_lr * m[i] / (sqrt_v + epsilon);
        }
      }
    };
    ParallelLaunch(task, input_elements_, kAdamBlock, this);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    gradient = gradient + input_elements_;
  }
}

void AdamCpuKernelMod::LaunchAdamNnacl(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &) {
  float *var = reinterpret_cast<float *>(inputs[kIndexVar]->addr);
  float *m = reinterpret_cast<float *>(inputs[kIndexM]->addr);
  float *v = reinterpret_cast<float *>(inputs[kIndexV]->addr);
  float *beta1_power = reinterpret_cast<float *>(inputs[kIndexBeta1Power]->addr);
  float *beta2_power = reinterpret_cast<float *>(inputs[kIndexBeta2Power]->addr);
  float *lr = reinterpret_cast<float *>(inputs[kIndexLr]->addr);
  float beta1 = reinterpret_cast<float *>(inputs[kIndexBeta1]->addr)[kScalarIndex];
  float beta2 = reinterpret_cast<float *>(inputs[kIndexBeta2]->addr)[kScalarIndex];
  float epsilon = reinterpret_cast<float *>(inputs[kIndexEpsilon]->addr)[kScalarIndex];
  float *gradient = reinterpret_cast<float *>(inputs[kIndexGrad]->addr);
  constexpr float ONE = 1.0;
  for (int64_t b = 0; b < batch_size_; b++) {
    float new_lr = lr[b] * std::sqrt(ONE - beta2_power[b]) / (ONE - beta1_power[b]);
    // multithreading
    auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
      int ret = AdamFp32(var, m, v, new_lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
      if (ret != NNACL_OK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', AdamFp32 failed. Error no: " << ret;
      }
    };
    ParallelLaunch(task, input_elements_, kAdamBlock, this);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    gradient = gradient + input_elements_;
  }
}

bool AdamCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  if (prim->HasAttr("use_locking")) {
    use_locking_ = GetValue<bool>(prim->GetAttr("use_locking"));
  }
  if (prim->HasAttr("use_nesterov")) {
    use_nesterov_ = GetValue<bool>(prim->GetAttr("use_nesterov"));
  }

  dtype_ = inputs.at(kIndex0)->GetDtype();
  kernel_name_ = prim->name();
  batch_rank_ = base_operator->get_batch_rank();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdamCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_elements_ = 0;

  std::vector<int64_t> var_shape = inputs[kIndexVar]->GetShapeVector();
  std::vector<int64_t> beta1_power_shape = inputs[kIndexBeta1Power]->GetShapeVector();
  std::vector<int64_t> beta2_power_shape = inputs[kIndexBeta2Power]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLr]->GetShapeVector();

  if (!IsSameShape(beta1_power_shape, beta2_power_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'beta1_power' and 'beta2_power' must be the same, "
                  << "but get the shapes of 'beta1_power': " << Vector2Str(beta1_power_shape)
                  << " and 'beta2_power': " << Vector2Str(beta2_power_shape);
    return KRET_RESIZE_FAILED;
  }

  if (batch_rank_ > 0 && lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;
  if (batch_rank_ > 1) {
    if (var_shape.size() < lr_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than 'lr_shape', but got the shape of 'var': "
                    << Vector2Str(var_shape) << " and 'lr_shape': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << Vector2Str(var_batch_shape) << " and the shape of 'lr': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
  }

  return KRET_OK;
}

template <typename T>
bool AdamCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdam<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

std::vector<std::pair<KernelAttr, AdamCpuKernelMod::AdamFunc>> AdamCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdamCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> AdamCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdamFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Adam, AdamCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
