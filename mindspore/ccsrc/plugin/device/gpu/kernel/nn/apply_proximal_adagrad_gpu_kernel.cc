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

#include "plugin/device/gpu/kernel/nn/apply_proximal_adagrad_gpu_kernel.h"
#include <algorithm>
#include "kernel/common_utils.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_proximal_adagrad_impl.cuh"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyProximalAdagradInputsNum = 6;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kLRIndex = 2;
constexpr size_t kL1Index = 3;
constexpr size_t kL2Index = 4;
constexpr size_t kGradIndex = 5;
}  // namespace

bool ApplyProximalAdagradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
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

int ApplyProximalAdagradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kApplyProximalAdagradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 6.";
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kAccIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kL1Index]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kL2Index]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << Vector2Str(accum_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'grad' must be the same as the shape of 'var', "
                     "but got the shape of 'grad': "
                  << Vector2Str(grad_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
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
  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
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

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
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

  return ret;
}

bool ApplyProximalAdagradGpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &workspace,
                                              const std::vector<kernel::AddressPtr> &outputs, void *cuda_stream) {
  kernel_func_(this, inputs, outputs, cuda_stream);
  return true;
}

template <typename T>
bool ApplyProximalAdagradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &, void *cuda_stream) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<T *>(inputs[kAccIndex]->addr);
  auto lr = reinterpret_cast<T *>(inputs[kLRIndex]->addr);
  auto l1 = reinterpret_cast<T *>(inputs[kL1Index]->addr);
  auto l2 = reinterpret_cast<T *>(inputs[kL2Index]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);

  auto status = CalApplyProximalAdagrad(input_elements_, batch_size_, lr, l1, l2, grad, var, accum, device_id_,
                                        reinterpret_cast<cudaStream_t>(cuda_stream));

  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ApplyProximalAdagradGpuKernelMod::KernelFunc>>
  ApplyProximalAdagradGpuKernelMod::func_list_ = {{KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutInRef(0, 0)
                                                     .AddOutInRef(1, 1),
                                                   &ApplyProximalAdagradGpuKernelMod::LaunchKernel<float>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutInRef(0, 0)
                                                     .AddOutInRef(1, 1),
                                                   &ApplyProximalAdagradGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> ApplyProximalAdagradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyProximalAdagrad, ApplyProximalAdagradGpuKernelMod);
};  // namespace kernel
}  // namespace mindspore
