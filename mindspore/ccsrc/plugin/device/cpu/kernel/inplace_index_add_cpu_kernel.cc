/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/inplace_index_add_cpu_kernel.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>

#include "include/common/thread_pool.h"
#include "mindspore/core/ops/inplace_index_add.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInplaceIndexAddInputsNum = 3;
constexpr size_t kInplaceIndexAddOutputsNum = 1;
}  // namespace

bool InplaceIndexAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::InplaceIndexAdd>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast InplaceIndexAdd ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  base_operator_ = base_operator;

  return true;
}

int InplaceIndexAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // Get input, output and attr info
  var_shape = inputs[kIndex0]->GetShapeVector();
  updates_shape = inputs[kIndex2]->GetShapeVector();
  indices_shape = inputs[kIndex1]->GetShapeVector();
  axis_ = GetValue<int64_t>(base_operator_->GetAttr(AXIS));
  return KRET_OK;
}

void InplaceIndexAddCpuKernelMod::CheckParams() {
  // Check dimension(var) = dimension(updates)
  if (var_shape.size() != updates_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'var' and 'updates' must have the same dimension, but got "
                      << var_shape.size() << " vs " << updates_shape.size() << ".";
  }
  // Check dimension(indices) = 1
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'indices' must have one dimension, but got "
                      << indices_shape.size() << ".";
  }
  // Check axis's value is valid
  auto var_dims = SizeToLong(var_shape.size());
  if (axis_ < -var_dims || axis_ >= var_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", 'axis' must be in range [" << -var_dims << ", " << var_dims
                      << "), but got " << axis_ << ".";
  }
  if (axis_ < 0) {
    axis_ += var_dims;
  }
  auto axis = LongToSize(axis_);
  // Check indices's size = updates.shape[axis]
  if (indices_shape[0] != updates_shape[axis]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", size of 'indices' must be the same as size of "
                         "'updates' in 'axis'th dimension, but got "
                      << indices_shape[0] << " vs " << updates_shape[axis] << ".";
  }
  // Check var.shape[i] = updates.shape[i], except i = axis
  var_nums = 1;
  updates_nums = 1;
  inner_size_ = 1;
  for (size_t i = 0; i < var_shape.size(); ++i) {
    if (var_shape[i] <= 0 || updates_shape[i] <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'var' shape[" << i << "] or 'updates' shape [" << i
                        << "] is invalid, which should > 0, but got " << var_shape[i] << " and " << updates_shape[i]
                        << ".";
    }
    if (i != axis && var_shape[i] != updates_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", the shape of 'var' and 'updates' must be same except the "
                           "'axis'th dimension, but got different values: "
                        << var_shape[i] << " vs " << updates_shape[i] << " in dimension " << i << ".";
    }
    var_nums *= LongToSize(var_shape[i]);
    updates_nums *= LongToSize(updates_shape[i]);
    if (i > axis) {
      inner_size_ *= LongToSize(var_shape[i]);
    }
  }
  x_axis_size_ = LongToSize(var_shape[axis_]);
  y_axis_size_ = LongToSize(updates_shape[axis_]);
}

template <typename T>
bool InplaceIndexAddCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInplaceIndexAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInplaceIndexAddOutputsNum, kernel_name_);
  auto *x = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *indices = reinterpret_cast<int32_t *>(inputs[kIndex1]->addr);
  auto *y = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  CheckParams();
  size_t x_axis_inner_size = x_axis_size_ * inner_size_;
  size_t y_axis_inner_size = y_axis_size_ * inner_size_;

  auto task1 = [&](const size_t start, const size_t end) {
    for (size_t i = start; i < end; ++i) {
      // calc idx_y in y.shape[axis]
      const size_t y_axis_idx = (i / inner_size_) % y_axis_size_;
      // calc idx_x in x.shape[axis]
      const size_t x_axis_idx = static_cast<size_t>(indices[y_axis_idx]);
      // only process add operation when idx_x is valid
      if (x_axis_idx < x_axis_size_) {
        const size_t x_outer_idx = i / y_axis_inner_size;
        const size_t x_inner_idx = i % inner_size_;
        const size_t x_idx = x_outer_idx * x_axis_inner_size + x_axis_idx * inner_size_ + x_inner_idx;
        x[x_idx] += y[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task1, updates_nums, this, &parallel_search_info_);

  auto task2 = [&](size_t start, size_t end) {
    size_t length = (end - start) * sizeof(T);
    auto ret = memcpy_s(output + start, length, x + start, length);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret << ".";
    }
  };
  ParallelLaunchAutoSearch(task2, var_nums, this, &parallel_search_info_);

  return true;
}

using KernelRunFunc = InplaceIndexAddCpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &InplaceIndexAddCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &InplaceIndexAddCpuKernelMod::LaunchKernel<uint8_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InplaceIndexAdd, InplaceIndexAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
