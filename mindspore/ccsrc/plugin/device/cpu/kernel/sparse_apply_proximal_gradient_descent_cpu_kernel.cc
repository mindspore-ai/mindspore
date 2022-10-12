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

#include "plugin/device/cpu/kernel/sparse_apply_proximal_gradient_descent_cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <memory>
#include <map>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyProximalGradientDescentInputsNum = 6;
constexpr size_t kSparseApplyProximalGradientDescentOutputsNum = 1;

using KernelRunFunc = SparseApplyProximalGradientDescentCpuKernelMod::KernelRunFunc;

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

bool SparseApplyProximalGradientDescentCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                          const std::vector<KernelTensorPtr> &inputs,
                                                          const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyProximalGradientDescentInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyProximalGradientDescentInputsNum
                  << ", but got " << inputs.size() << ".";
    return false;
  }
  if (outputs.size() != kSparseApplyProximalGradientDescentOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be "
                  << kSparseApplyProximalGradientDescentOutputsNum << ", but got " << outputs.size() << ".";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyProximalGradientDescentCpuKernelMod::ResetResouce() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyProximalGradientDescentCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                           const std::vector<KernelTensorPtr> &inputs,
                                                           const std::vector<KernelTensorPtr> &outputs,
                                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResouce();
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != static_cast<int>(KRET_OK)) {
    return ret;
  }
  enum input_index : size_t { Var_no, Alpha_no, L1_no, L2_no, Grad_no, Indices_no };
  auto var_shape = inputs[static_cast<size_t>(Var_no)]->GetShapeVector();
  auto alpha_shape = inputs[static_cast<size_t>(Alpha_no)]->GetShapeVector();
  auto l1_shape = inputs[static_cast<size_t>(L1_no)]->GetShapeVector();
  auto l2_shape = inputs[static_cast<size_t>(L2_no)]->GetShapeVector();
  auto grad_shape = inputs[static_cast<size_t>(Grad_no)]->GetShapeVector();
  auto indices_shape = inputs[static_cast<size_t>(Indices_no)]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, var must be at least 1D.";
  } else {
    var_first_dim_size_ = LongToSize(var_shape[0]);
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, rank(grad) should be same as rank(var), but "
                         "got rank(grad): "
                      << grad_shape.size() << ", rank(var): " << var_shape.size() << ".";
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, the shape of var and grad must equal in dimension "
                        << i << ".";
    }
    var_outer_dim_size_ *= LongToSize(var_shape[i]);
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, indices must be 1D, but got " << indices_shape.size()
                      << "D.";
  }
  indices_size_ = LongToSize(indices_shape[0]);
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, grad.shape[0] must be equal to indices.shape[0], but "
                         "got grad.shape[0]: "
                      << grad_shape[0] << " indices.shape[0]: " << indices_size_ << ".";
  }
  if (!alpha_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, alpha is not a scalar, got shape: "
                      << Vector2Str(alpha_shape) << ".";
  }
  if (!l1_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, l1 is not a scalar, got shape: "
                      << Vector2Str(l1_shape) << ".";
  }
  if (!l2_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, l2 is not a scalar, got shape: "
                      << Vector2Str(l2_shape) << ".";
  }
  return static_cast<int>(KRET_OK);
}

template <typename I, typename T>
bool SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel(
  const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
  const std::vector<kernel::AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyProximalGradientDescentInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyProximalGradientDescentOutputsNum, kernel_name_);

  auto var = static_cast<T *>(inputs[0]->addr);
  auto grad = static_cast<T *>(inputs[4]->addr);
  auto indices = static_cast<I *>(inputs[5]->addr);
  auto alpha_scalar = static_cast<T *>(inputs[1]->addr)[0];
  auto l1_scalar = static_cast<T *>(inputs[2]->addr)[0];
  auto l2_scalar = static_cast<T *>(inputs[3]->addr)[0];
  auto output = static_cast<T *>(outputs[0]->addr);

  for (size_t i = 0; i < indices_size_; i++) {
    I index = indices[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION)
        << "For SparseApplyProximalGradientDescent, values in indices should be [0, var.shape[0]), but got " << index
        << ".";
    }
    size_t start_index = var_outer_dim_size_ * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      auto learning_rate = alpha_scalar;
      auto prox_v = var[j];
      prox_v -= grad[k] * learning_rate;
      if (l1_scalar > static_cast<T>(0.0)) {
        var[j] = static_cast<T>(Sign(static_cast<double>(prox_v))) *
                 static_cast<T>(std::fmax(std::fabs(static_cast<double>(prox_v)) -
                                            static_cast<double>(learning_rate) * static_cast<double>(l1_scalar),
                                          static_cast<double>(0.0))) /
                 (static_cast<T>(1.0) + l2_scalar * learning_rate);
      } else {
        var[j] = static_cast<T>(prox_v) / (static_cast<T>(1.0) + l2_scalar * learning_rate);
      }
    }
  }

  auto copy_size = var_first_dim_size_ * var_outer_dim_size_ * sizeof(T);
  auto ret = memcpy_s(output, copy_size, var, copy_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For SparseApplyProximalGradientDescent, memcpy_s error, errorno: " << ret << ".";
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyProximalGradientDescentCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list_ = {
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int8, Int32, Int8),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int16, Int32, Int16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int32, Int64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, UInt8, Int32, UInt8),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, UInt16, Int32, UInt16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, UInt32, Int32, UInt32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, UInt64, Int32, UInt64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Float16, Int32, Float16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Float32, Int32, Float32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Float64, Int32, Float64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int32_t, double>},
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int8, Int64, Int8),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int16, Int64, Int16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int64, Int32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, UInt8, Int64, UInt8),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, UInt16, Int64, UInt16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, UInt32, Int64, UInt32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, UInt64, Int64, UInt64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Float16, Int64, Float16),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Float32, Int64, Float32),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Float64, Int64, Float64),
     &SparseApplyProximalGradientDescentCpuKernelMod::LaunchKernel<int64_t, double>}};
  return func_list_;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyProximalGradientDescent,
                      SparseApplyProximalGradientDescentCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
