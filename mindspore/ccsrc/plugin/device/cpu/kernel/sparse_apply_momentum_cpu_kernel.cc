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

#include "plugin/device/cpu/kernel/sparse_apply_momentum_cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <memory>
#include <map>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyMomentumInputsNum = 6;
constexpr size_t kSparseApplyMomentumOutputsNum = 1;

using KernelRunFunc = SparseApplyMomentumCpuKernelMod::KernelRunFunc;

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7) \
  KernelAttr()                                 \
    .AddInputAttr(kNumberType##t1)             \
    .AddInputAttr(kNumberType##t2)             \
    .AddInputAttr(kNumberType##t3)             \
    .AddInputAttr(kNumberType##t4)             \
    .AddInputAttr(kNumberType##t5)             \
    .AddInputAttr(kNumberType##t6)             \
    .AddOutputAttr(kNumberType##t7)
}  // namespace

bool SparseApplyMomentumCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyMomentumInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyMomentumInputsNum
                  << ", but got " << inputs.size() << ".";
    return false;
  }
  if (outputs.size() != kSparseApplyMomentumOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be " << kSparseApplyMomentumOutputsNum
                  << ", but got " << outputs.size() << ".";
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::SparseApplyMomentum>(base_operator->GetPrim());
  use_nesterov_ = kernel_ptr->get_use_nesterov();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyMomentumCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
  use_nesterov_ = false;
}

int SparseApplyMomentumCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  enum input_index : size_t { Var_no, Accum_no, Lr_no, Grad_no, Indices_no, Momentum_no };
  auto var_shape = inputs[static_cast<size_t>(Var_no)]->GetShapeVector();
  auto accum_shape = inputs[static_cast<size_t>(Accum_no)]->GetShapeVector();
  auto lr_shape = inputs[static_cast<size_t>(Lr_no)]->GetShapeVector();
  auto grad_shape = inputs[static_cast<size_t>(Grad_no)]->GetShapeVector();
  auto indices_shape = inputs[static_cast<size_t>(Indices_no)]->GetShapeVector();
  auto momentum_shape = inputs[static_cast<size_t>(Momentum_no)]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, var must be at least 1D.";
  } else {
    var_first_dim_size_ = LongToSize(var_shape[0]);
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, rank(grad) should be same as rank(var), but got rank(grad): "
                      << grad_shape.size() << ", rank(var): " << var_shape.size() << ".";
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, var and accum should have the same shape.";
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For SparseApplyMomentum, the shape of var and grad must equal in dimension " << i << ".";
    }
    var_outer_dim_size_ *= LongToSize(var_shape[i]);
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, indices must be 1D, but got " << indices_shape.size() << "D.";
  }
  indices_size_ = LongToSize(indices_shape[0]);
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(EXCEPTION)
      << "For SparseApplyMomentum, grad.shape[0] must be equal to indices.shape[0], but got grad.shape[0]: "
      << grad_shape[0] << ", indices.shape[0]: " << indices_size_ << ".";
  }
  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, lr is not a scalar, got shape: " << Vector2Str(lr_shape) << ".";
  }
  if (!momentum_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, momentum is not a scalar, got shape: " << Vector2Str(momentum_shape)
                      << ".";
  }
  return static_cast<int>(KRET_OK);
}

template <typename I, typename T>
bool SparseApplyMomentumCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyMomentumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyMomentumOutputsNum, kernel_name_);

  auto var = static_cast<T *>(inputs[0]->addr);
  auto accum = static_cast<T *>(inputs[1]->addr);
  auto grad = static_cast<T *>(inputs[3]->addr);
  auto indices = static_cast<I *>(inputs[4]->addr);
  auto lr_scalar = static_cast<T *>(inputs[2]->addr)[0];
  auto momentum_scalar = static_cast<T *>(inputs[5]->addr)[0];
  auto output = static_cast<T *>(outputs[0]->addr);

  for (size_t i = 0; i < indices_size_; ++i) {
    I index = indices[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "For SparseApplyMomentum, values in indices should be [0, var.shape[0]), but got " << index
                        << ".";
    }
    size_t start_index = var_outer_dim_size_ * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      accum[j] = accum[j] * momentum_scalar + grad[k];
      if (use_nesterov_) {
        var[j] -= lr_scalar * grad[k] + lr_scalar * momentum_scalar * accum[j];
      } else {
        var[j] -= lr_scalar * accum[j];
      }
    }
  }

  size_t copy_size = var_first_dim_size_ * var_outer_dim_size_ * sizeof(T);
  auto ret = memcpy_s(output, copy_size, var, copy_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For SparseApplyMomentum, memcpy_s error, errorno: " << ret << ".";
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyMomentumCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list_ = {
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int32, Int8, Int8),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int32, Int16, Int16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int32, Int64, Int64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, Int32, UInt8, UInt8),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, Int32, UInt16, UInt16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, Int32, UInt32, UInt32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, Int32, UInt64, UInt64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Int32, Float16, Float16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Int32, Float32, Float32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Int32, Float64, Float64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int32_t, double>},
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int64, Int8, Int8),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int64, Int16, Int16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int64, Int32, Int32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, Int64, UInt8, UInt8),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, Int64, UInt16, UInt16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, Int64, UInt32, UInt32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, Int64, UInt64, UInt64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Int64, Float16, Float16),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Int64, Float32, Float32),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Int64, Float64, Float64),
     &SparseApplyMomentumCpuKernelMod::LaunchKernel<int64_t, double>}};
  return func_list_;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyMomentum, SparseApplyMomentumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
