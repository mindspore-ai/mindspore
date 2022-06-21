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

#include "plugin/device/cpu/kernel/sparse_apply_adagrad_da_cpu_kernel.h"

#include <algorithm>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyAdagradDAInputsNum = 9;
constexpr size_t kSparseApplyAdagradDAOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) \
  KernelAttr()                                              \
    .AddInputAttr(kNumberType##t1)                          \
    .AddInputAttr(kNumberType##t2)                          \
    .AddInputAttr(kNumberType##t3)                          \
    .AddInputAttr(kNumberType##t4)                          \
    .AddInputAttr(kNumberType##t5)                          \
    .AddInputAttr(kNumberType##t6)                          \
    .AddInputAttr(kNumberType##t7)                          \
    .AddInputAttr(kNumberType##t8)                          \
    .AddInputAttr(kNumberType##t9)                          \
    .AddOutputAttr(kNumberType##t10)
}  // namespace

void SparseApplyAdagradDACpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  enum input_index : size_t { Var_no, Ga_no, Gs_no, Grad_no, Indices_no, Lr_no, L1_no, L2_no, Global_step_no };
  ShapeVector var_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Var_no);
  ShapeVector grad_accum_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Ga_no);
  ShapeVector grad_square_accum_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Gs_no);
  ShapeVector grad_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Grad_no);
  ShapeVector indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Indices_no);
  ShapeVector lr_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Lr_no);
  ShapeVector l1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, L1_no);
  ShapeVector l2_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, L2_no);
  ShapeVector global_step_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, Global_step_no);
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var must be at least 1D.";
  } else {
    var_first_dim_size_ = var_shape[0];
  }
  if (!IsSameShape(var_shape, grad_accum_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad_accum should have the same shape.";
  }
  if (!IsSameShape(var_shape, grad_square_accum_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad_square_accum shape should have the same shape.";
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad should have the same shape size.";
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, the shape of var and grad must equal in dimension " << i << ".";
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, indices must be 1D, but got " << indices_shape.size() << ".";
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(EXCEPTION)
      << "For SparseApplyAdagradDA, grad.shape[0] must be equal to indices.shape[0], but got grad_shape[0]: "
      << grad_shape[0] << ", indices_shape[0]: " << indices_size_ << ".";
  }
  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, lr is not a scalar, got shape: " << Vector2Str(lr_shape) << ".";
  }
  if (!l1_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, l1 is not a scalar, got shape: " << Vector2Str(l1_shape) << ".";
  }
  if (!l2_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, l2 is not a scalar, got shape: " << Vector2Str(l2_shape) << ".";
  }
  if (!global_step_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, global_step is not a scalar, got shape: "
                      << Vector2Str(global_step_shape) << ".";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, this kernel data type are not supported: " << kernel_attr << ".";
  }
  kernel_func_ = func_list_[index].second;
}

template <typename I, typename T>
bool SparseApplyAdagradDACpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyAdagradDAOutputsNum, kernel_name_);

  auto var = reinterpret_cast<T *>(inputs[0]->addr);
  auto ga = reinterpret_cast<T *>(inputs[1]->addr);
  auto da = reinterpret_cast<T *>(inputs[2]->addr);
  auto g = reinterpret_cast<T *>(inputs[3]->addr);
  auto indices = reinterpret_cast<I *>(inputs[4]->addr);
  auto lr_scalar = reinterpret_cast<T *>(inputs[5]->addr)[0];
  auto l1_scalar = reinterpret_cast<T *>(inputs[6]->addr)[0];
  auto l2_scalar = reinterpret_cast<T *>(inputs[7]->addr)[0];
  int64_t global_step_scalar_int64 = reinterpret_cast<int64_t *>(inputs[8]->addr)[0];
  T global_step_scalar = static_cast<T>(global_step_scalar_int64);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto gs_lr = global_step_scalar * lr_scalar;
  for (size_t i = 0; i < indices_size_; ++i) {
    I index = indices[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, values in indices should be [0, var.shape[0]), but got " << index
                        << ".";
    }
    size_t start_index = var_outer_dim_size_ * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      ga[j] = ga[j] + g[k];
      da[j] = da[j] + g[k] * g[k];
      if (l1_scalar > static_cast<T>(0.0)) {
        var[j] =
          static_cast<T>(-1.0) * (T)Sign(static_cast<double>(ga[j])) *
          (T)std::fmax(static_cast<double>(((T)std::fabs(static_cast<double>(ga[j])) / global_step_scalar) - l1_scalar),
                       static_cast<double>(0.0)) /
          (l2_scalar + (T)std::sqrt(static_cast<double>(da[j])) / gs_lr);
      } else {
        var[j] = static_cast<T>(-1.0) * (ga[j] / global_step_scalar) /
                 (l2_scalar + (T)std::sqrt(static_cast<double>(da[j])) / gs_lr);
      }
    }
  }
  size_t copy_size = var_first_dim_size_ * var_outer_dim_size_ * sizeof(T);
  auto ret = memcpy_s(output, copy_size, var, copy_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, memcpy_s error, errorno" << ret << ".";
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseApplyAdagradDACpuKernelMod::SparseApplyAdagradDAFunc>>
  SparseApplyAdagradDACpuKernelMod::func_list_ = {
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int32, Int8, Int8, Int8, Int64, Int8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int32, Int16, Int16, Int16, Int64, Int16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int64, Int32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int32, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, Int32, UInt8, UInt8, UInt8, Int64, UInt8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, Int32, UInt16, UInt16, UInt16, Int64, UInt16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, Int32, UInt32, UInt32, UInt32, Int64, UInt32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, Int32, UInt64, UInt64, UInt64, Int64, UInt64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Int32, Float16, Float16, Float16, Int64, Float16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Int32, Float32, Float32, Float32, Int64, Float32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Int32, Float64, Float64, Float64, Int64, Float64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, double>},
    {ADD_KERNEL(Int8, Int8, Int8, Int8, Int64, Int8, Int8, Int8, Int64, Int8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {ADD_KERNEL(Int16, Int16, Int16, Int16, Int64, Int16, Int16, Int16, Int64, Int16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int64, Int32, Int32, Int32, Int64, Int32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, Int64, UInt8, UInt8, UInt8, Int64, UInt8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, uint8_t>},
    {ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, Int64, UInt16, UInt16, UInt16, Int64, UInt16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, uint16_t>},
    {ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, Int64, UInt32, UInt32, UInt32, Int64, UInt32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, uint32_t>},
    {ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, Int64, UInt64, UInt64, UInt64, Int64, UInt64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, uint64_t>},
    {ADD_KERNEL(Float16, Float16, Float16, Float16, Int64, Float16, Float16, Float16, Int64, Float16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, float16>},
    {ADD_KERNEL(Float32, Float32, Float32, Float32, Int64, Float32, Float32, Float32, Int64, Float32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, float>},
    {ADD_KERNEL(Float64, Float64, Float64, Float64, Int64, Float64, Float64, Float64, Int64, Float64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, double>}};

std::vector<KernelAttr> SparseApplyAdagradDACpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseApplyAdagradDAFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyAdagradDA, SparseApplyAdagradDACpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
