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

#include "mindspore/core/ops/sparse_apply_adagrad.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "plugin/device/gpu/kernel/nn/sparse_apply_adagrad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyAdagradInputsNum = 4;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kGradIndex = 2;
constexpr size_t kIndicesIndex = 3;
}  // namespace

bool SparseApplyAdagradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimSparseApplyAdagrad->name()) {
    MS_LOG(ERROR) << "For 'SparseApplyAdagrad', the kernel name must be 'SparseApplyAdagrad', but got " << kernel_name_;
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyAdagrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "SparseApplyAdagrad ops failed!";
    return false;
  }
  lr_ = kernel_ptr->get_lr();
  update_slots_ = kernel_ptr->get_update_slots();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
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

int SparseApplyAdagradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  if (input_size_list_.size() != kSparseApplyAdagradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 4 but got " << input_size_list_.size();
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kAccIndex]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  std::vector<int64_t> indices_shape = inputs[kIndicesIndex]->GetShapeVector();
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
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'grad' must be the same as the dimension of "
                     "'var', but got the dimension of 'grad': "
                  << grad_shape.size() << " and the dimension of 'var': " << var_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var' and 'grad' must be equal in dimension i=" << i
                    << ", but got 'var_shape[i]': " << var_shape[i] << " and 'grad_shape[i]': " << grad_shape[i];
      return KRET_RESIZE_FAILED;
    }
  }
  if (indices_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' must be a 1-D vector, but got "
                  << indices_shape.size() << "-D.";
    return KRET_RESIZE_FAILED;
  }
  auto indices_size = indices_shape[0];
  if (grad_shape[0] != SizeToLong(indices_size)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the first dimension value of 'grad' must be equal to "
                     "the first dimension value of 'indices', but got the first dimension value of 'grad': "
                  << grad_shape[0] << ", and the first dimension value of 'indices': " << indices_size;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = input_size_list_[0] / unit_size_;
  return ret;
}

template <typename T, typename S>
bool SparseApplyAdagradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<T *>(inputs[kAccIndex]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  auto indices = reinterpret_cast<S *>(inputs[kIndicesIndex]->addr);
  auto var_out = reinterpret_cast<T *>(outputs[kVarIndex]->addr);
  auto accum_out = reinterpret_cast<T *>(outputs[kAccIndex]->addr);

  auto status = CalSparseApplyAdagrad(input_elements_, sizeof(S) / sizeof(int), lr_, update_slots_, grad, indices, var,
                                      accum, var_out, accum_out, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseApplyAdagradGpuKernelMod::SparseApplyAdagradFunc>>
  SparseApplyAdagradGpuKernelMod::func_list_ = {{KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutInRef(0, 0)
                                                   .AddOutInRef(1, 1),
                                                 &SparseApplyAdagradGpuKernelMod::LaunchKernel<float, int>},
                                                {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutInRef(0, 0)
                                                   .AddOutInRef(1, 1),
                                                 &SparseApplyAdagradGpuKernelMod::LaunchKernel<half, int>}};

std::vector<KernelAttr> SparseApplyAdagradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseApplyAdagradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseApplyAdagrad, SparseApplyAdagradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
