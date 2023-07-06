/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <map>
#include <functional>
#include <algorithm>

#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/apply_adam_with_amsgrad_v2_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "mindspore/core/ops/apply_adam_with_amsgradv2.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyAdamWithAmsgradV2InputsNum = 11;
constexpr size_t kApplyAdamWithAmsgradV2OutputsNum = 4;
constexpr size_t kScalarIndex = 0;
}  // namespace

bool ApplyAdamWithAmsgradV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ApplyAdamWithAmsgradV2>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();

  if (inputs.size() != kApplyAdamWithAmsgradV2InputsNum || outputs.size() != kApplyAdamWithAmsgradV2OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size should be "
                  << kApplyAdamWithAmsgradV2InputsNum << " and " << kApplyAdamWithAmsgradV2OutputsNum << ", but got "
                  << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel data type: " << kernel_attr << " is not supported.";
    return false;
  }

  return true;
}

int ApplyAdamWithAmsgradV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }

  std::vector<int64_t> var_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kIndex1]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kIndex2]->GetShapeVector();
  std::vector<int64_t> vhat_shape = inputs[kIndex3]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndex6]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndex10]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, m_shape) || !IsSameShape(var_shape, v_shape) || !IsSameShape(var_shape, vhat_shape) ||
      !IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'm/v/vhat/grad/var' must be the same, "
                  << "but get the shapes of 'm': " << m_shape << ", 'v': " << v_shape << ", 'vhat': " << vhat_shape
                  << ", 'grad': " << grad_shape << " and 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), int64_t(1), std::multiplies<int64_t>());
  }

  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  int64_t temp_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
  input_elements_ = static_cast<size_t>(temp_elements_ / batch_size_);

  return 0;
}

template <typename T>
void ApplyAdamWithAmsgradV2CpuKernelMod::LaunchApplyAdamWithAmsgradV2(const std::vector<AddressPtr> &inputs,
                                                                      const std::vector<AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  T *vhat = reinterpret_cast<T *>(inputs[kIndex3]->addr);
  T *beta1_power = reinterpret_cast<T *>(inputs[kIndex4]->addr);
  T *beta2_power = reinterpret_cast<T *>(inputs[kIndex5]->addr);
  T *lr = reinterpret_cast<T *>(inputs[kIndex6]->addr);
  T *beta1 = reinterpret_cast<T *>(inputs[kIndex7]->addr);
  T *beta2 = reinterpret_cast<T *>(inputs[kIndex8]->addr);
  T *epsilon = reinterpret_cast<T *>(inputs[kIndex9]->addr);
  T *gradient = reinterpret_cast<T *>(inputs[kIndex10]->addr);

  T ONE = static_cast<T>(1.0);
  for (int64_t b = 0; b < batch_size_; b++) {
    // multithreading
    T new_lr = lr[b] * static_cast<T>(std::sqrt(static_cast<double>(ONE - beta2_power[b]))) / (ONE - beta1_power[b]);
    auto task = [this, &var, &m, &v, &vhat, &gradient, new_lr, &beta1, &beta2, &epsilon](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] += (gradient[i] - m[i]) * (one - *beta1);
        v[i] += (gradient[i] * gradient[i] - v[i]) * (one - *beta2);
        vhat[i] = std::max(vhat[i], v[i]);
        T sqrt_vhat = static_cast<T>(std::sqrt(static_cast<double>(vhat[i])));
        var[i] -= new_lr * m[i] / (sqrt_vhat + *epsilon);
      }
    };
    ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    vhat = vhat + input_elements_;
    gradient = gradient + input_elements_;
  }
}

bool ApplyAdamWithAmsgradV2CpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  if (inputs[kIndex0]->size != inputs[kIndex1]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'm' and 'var' should be same, but got the memory size of 'm': "
                      << inputs[kIndex1]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex0]->size != inputs[kIndex2]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'v' and 'var' should be same, but got the memory size of 'v': "
                      << inputs[kIndex2]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex0]->size != inputs[kIndex3]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'vhat' and 'var' should be same, but got the size of 'vhat': "
                      << inputs[kIndex3]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex0]->size != inputs[kIndex10]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'gradient' and 'var' should be same, but got "
                         "the memory size of 'gradient': "
                      << inputs[kIndex10]->size << " and 'var': " << inputs[kIndex0]->size;
  }

  if (dtype_ == kNumberTypeFloat64) {
    LaunchApplyAdamWithAmsgradV2<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchApplyAdamWithAmsgradV2<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchApplyAdamWithAmsgradV2<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be float16, float32 or float64, but "
                      << "get " << TypeIdToType(dtype_)->ToString();
  }

  return true;
}

std::vector<KernelAttr> ApplyAdamWithAmsgradV2CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutInRef(0, 0)
                                                   .AddOutInRef(1, 1)
                                                   .AddOutInRef(2, 2)
                                                   .AddOutInRef(3, 3),
                                                 KernelAttr()
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
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutInRef(0, 0)
                                                   .AddOutInRef(1, 1)
                                                   .AddOutInRef(2, 2)
                                                   .AddOutInRef(3, 3),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutInRef(0, 0)
                                                   .AddOutInRef(1, 1)
                                                   .AddOutInRef(2, 2)
                                                   .AddOutInRef(3, 3)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
