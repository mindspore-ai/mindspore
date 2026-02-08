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

#include "kernel/gpu/cuda/nn/apply_proximal_gradient_descent_gpu_kernel.h"
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/apply_proximal_gradient_descent_impl.cuh"
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

bool ApplyProximalGradientDescentGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
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

int ApplyProximalGradientDescentGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.size() != kApplyProximalGradientDescentInputsNum) {
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
                  << delta_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!alpha_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'alpha' must be a scalar,and dimension of 'alpha' must be 0,but got the dimension of 'alpha': "
                  << alpha_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l1_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l1' must be a scalar,and dimension of 'l1' must be 0,but got the dimension of 'l1': "
                  << l1_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l2_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l2' must be a scalar,and dimension of 'l2' must be 0,but got the dimension of 'l2': "
                  << l2_shape;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = inputs[0]->size() / unit_size_;
  return ret;
}

template <typename T>
bool ApplyProximalGradientDescentGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &workspace,
                                                            const std::vector<KernelTensor *> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->device_ptr());
  auto alpha = reinterpret_cast<T *>(inputs[kAlphaIndex]->device_ptr());
  auto l1 = reinterpret_cast<T *>(inputs[kL1Index]->device_ptr());
  auto l2 = reinterpret_cast<T *>(inputs[kL2Index]->device_ptr());
  auto delta = reinterpret_cast<T *>(inputs[kDeltaIndex]->device_ptr());
  auto output = reinterpret_cast<T *>(outputs[kOutputIndex]->device_ptr());

  auto status = CalApplyProximalGradientDescent(input_elements_, var, alpha, l1, l2, delta, output, device_id_,
                                                reinterpret_cast<cudaStream_t>(cuda_stream_));

  CHECK_CUDA_STATUS(status, kernel_name_);
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
