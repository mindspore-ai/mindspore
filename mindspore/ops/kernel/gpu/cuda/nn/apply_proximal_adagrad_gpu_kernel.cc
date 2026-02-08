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

#include "kernel/gpu/cuda/nn/apply_proximal_adagrad_gpu_kernel.h"
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/apply_proximal_adagrad_impl.cuh"
#include "ops_utils/op_utils.h"
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

bool ApplyProximalAdagradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  batch_rank_ = ops::get_batch_rank(primitive_);
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

int ApplyProximalAdagradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.size() != kApplyProximalAdagradInputsNum) {
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
                  << accum_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'grad' must be the same as the shape of 'var', "
                     "but got the shape of 'grad': "
                  << grad_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l1_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l1', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and the shape of 'l1': " << l1_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l2_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l2', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and the shape of 'l2': " << l2_shape;
    return KRET_RESIZE_FAILED;
  }
  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and 'batch_rank': " << batch_rank_;
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
                    << var_shape << " and 'lr_shape': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << var_batch_shape << " and the shape of 'lr': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
  }

  return ret;
}

bool ApplyProximalAdagradGpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<kernel::KernelTensor *> &workspace,
                                              const std::vector<kernel::KernelTensor *> &outputs, void *cuda_stream) {
  kernel_func_(this, inputs, outputs, cuda_stream);
  return true;
}

template <typename T>
bool ApplyProximalAdagradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &, void *cuda_stream) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->device_ptr());
  auto accum = reinterpret_cast<T *>(inputs[kAccIndex]->device_ptr());
  auto lr = reinterpret_cast<T *>(inputs[kLRIndex]->device_ptr());
  auto l1 = reinterpret_cast<T *>(inputs[kL1Index]->device_ptr());
  auto l2 = reinterpret_cast<T *>(inputs[kL2Index]->device_ptr());
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->device_ptr());

  auto status = CalApplyProximalAdagrad(input_elements_, batch_size_, lr, l1, l2, grad, var, accum, device_id_,
                                        reinterpret_cast<cudaStream_t>(cuda_stream));

  CHECK_CUDA_STATUS(status, kernel_name_);
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
