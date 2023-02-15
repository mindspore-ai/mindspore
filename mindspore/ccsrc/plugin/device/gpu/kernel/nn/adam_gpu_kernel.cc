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
#include "plugin/device/gpu/kernel/nn/adam_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
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
}  // namespace

bool AdamGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  if (prim->HasAttr("use_locking")) {
    use_locikng_ = GetValue<bool>(prim->GetAttr("use_locking"));
  }
  if (prim->HasAttr("use_nesterov")) {
    use_nesterov_ = GetValue<bool>(prim->GetAttr("use_nesterov"));
  }

  kernel_name_ = prim->name();
  batch_rank_ = base_operator->get_batch_rank();
  constexpr size_t input_num = 10;
  constexpr size_t output_num = 3;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdamGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool AdamGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *m = GetDeviceAddress<T>(inputs, kIndex1);
  T *v = GetDeviceAddress<T>(inputs, kIndex2);
  T *beta1_power = GetDeviceAddress<T>(inputs, kIndex3);
  T *beta2_power = GetDeviceAddress<T>(inputs, kIndex4);
  T *learning_rate = GetDeviceAddress<T>(inputs, kIndex5);
  T *beta1 = GetDeviceAddress<T>(inputs, kIndex6);
  T *beta2 = GetDeviceAddress<T>(inputs, kIndex7);
  T *epsilon = GetDeviceAddress<T>(inputs, kIndex8);
  T *gradient = GetDeviceAddress<T>(inputs, kIndex9);
  ApplyAdam(input_elements_, batch_size_, gradient, beta1_power, beta2_power, learning_rate, beta1, beta2, epsilon,
            variable, m, v, use_nesterov_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, AdamGpuKernelMod::AdamLaunchFunc>> AdamGpuKernelMod::func_list_ = {
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
   &AdamGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
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
     .AddOutputAttr(kNumberTypeFloat16),
   &AdamGpuKernelMod::LaunchKernel<half>},
};
std::vector<KernelAttr> AdamGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdamGpuKernelMod::AdamLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Adam, AdamGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
