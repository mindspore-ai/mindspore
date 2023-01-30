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

#include "plugin/device/gpu/kernel/nn/apply_proximal_gradient_descent_gpu_kernel.h"
#include <algorithm>
#include "kernel/common_utils.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_proximal_gradient_descent_impl.cuh"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyProximalGradientDescentInputsNum = 5;
constexpr size_t kApplyProximalGradientDescentOutputsNum = 1;
constexpr size_t kVarIndex = 0;
constexpr size_t kAlphaIndex = 1;
constexpr size_t kL1Index = 2;
constexpr size_t kL2Index = 3;
constexpr size_t kDeltaIndex = 4;
constexpr size_t kOutputIndex = 0;
}  // namespace

bool ApplyProximalGradientDescentGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int ApplyProximalGradientDescentGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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

  if (!alpha_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'alpha' must be a scalar,and dimension of 'alpha' must be 0,but got the dimension of 'alpha': "
                  << Vector2Str(alpha_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!l1_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l1' must be a scalar,and dimension of 'l1' must be 0,but got the dimension of 'l1': "
                  << Vector2Str(l1_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!l2_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l2' must be a scalar,and dimension of 'l2' must be 0,but got the dimension of 'l2': "
                  << Vector2Str(l2_shape);
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = input_size_list_[0] / unit_size_;
  return ret;
}

template <typename T>
bool ApplyProximalGradientDescentGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                            const std::vector<AddressPtr> &workspace,
                                                            const std::vector<AddressPtr> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto alpha = reinterpret_cast<T *>(inputs[kAlphaIndex]->addr);
  auto l1 = reinterpret_cast<T *>(inputs[kL1Index]->addr);
  auto l2 = reinterpret_cast<T *>(inputs[kL2Index]->addr);
  auto delta = reinterpret_cast<T *>(inputs[kDeltaIndex]->addr);
  auto output = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);

  CalApplyProximalGradientDescent(input_elements_, var, alpha, l1, l2, delta, output, device_id_,
                                  reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

std::vector<std::pair<KernelAttr, ApplyProximalGradientDescentGpuKernelMod::KernelFunc>>
  ApplyProximalGradientDescentGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &ApplyProximalGradientDescentGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutInRef(0, 0),
     &ApplyProximalGradientDescentGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutInRef(0, 0),
     &ApplyProximalGradientDescentGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ApplyProximalGradientDescentGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyProximalGradientDescent, ApplyProximalGradientDescentGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
