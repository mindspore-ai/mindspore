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

#include "plugin/device/cpu/kernel/index_add_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include <unordered_set>

#include "mindspore/core/ops/index_add.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexAddInputsNum = 3;
constexpr size_t kIndexAddOutputsNum = 1;

bool HasDuplicateIndex(const int32_t *indices, size_t len) {
  MS_EXCEPTION_IF_NULL(indices);
  std::unordered_set<int32_t> unique_idx;
  for (size_t i = 0; i < len; ++i) {
    if (unique_idx.find(indices[i]) != unique_idx.end()) {
      return true;
    }
    (void)unique_idx.insert(indices[i]);
  }
  return false;
}

size_t CalcSizePerThread(size_t total_block) {
  size_t pool_thread_num = GetActorMgrInnerThreadPool()->GetKernelThreadNum();
  pool_thread_num = pool_thread_num == 0 ? 1 : pool_thread_num;
  size_t block_num = (total_block + pool_thread_num - 1) / pool_thread_num;
  return block_num;
}
}  // namespace

bool IndexAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::IndexAdd>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast IndexAdd ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  axis_ = kernel_ptr->get_axis();
  if (inputs.size() != kIndexAddInputsNum || outputs.size() != kIndexAddOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be " << kIndexAddInputsNum
                  << " and " << kIndexAddOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int IndexAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // Get input, output and attr info
  x_shape_ = inputs[kIndex0]->GetShapeVector();
  y_shape_ = inputs[kIndex2]->GetShapeVector();
  indices_shape_ = inputs[kIndex1]->GetShapeVector();
  return KRET_OK;
}

void IndexAddCpuKernelMod::CheckParams() {
  // Check dimension(x) = dimension(y)
  if (x_shape_.size() != y_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'x' and 'y' must have the same dimension, but got "
                      << x_shape_.size() << " vs " << y_shape_.size();
  }
  // Check dimension(indices) = 1
  if (indices_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'indices' must have one dimension, but got "
                      << indices_shape_.size();
  }
  // Check axis's value is valid
  auto x_rank = SizeToLong(x_shape_.size());
  if (axis_ < -x_rank || axis_ >= x_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", 'axis' must be in range [" << -x_rank << ", " << x_rank
                      << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += x_rank;
  }
  auto axis = LongToSize(axis_);
  // Check indices's size = y.shape[axis]
  if (indices_shape_[0] != y_shape_[axis]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", size of 'indices' must be the same as size of 'y' in 'axis'th dimension, but got "
                      << indices_shape_[0] << " vs " << y_shape_[axis];
  }
  // Check x.shape[i] = y.shape[i], except i = axis
  x_nums_ = 1;
  y_nums_ = 1;
  inner_size_ = 1;
  outer_size_ = 1;
  for (size_t i = 0; i < x_shape_.size(); ++i) {
    if (x_shape_[i] <= 0 || y_shape_[i] <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'x' shape[" << i << "] or 'y' shape [" << i
                        << "] is invalid, which should > 0, but got " << x_shape_[i] << " and " << y_shape_[i];
    }
    if (i != axis && x_shape_[i] != y_shape_[i]) {
      MS_LOG(EXCEPTION)
        << "For '" << kernel_name_
        << ", the shape of 'x' and 'y' must be same except the 'axis'th dimension, but got different values: "
        << x_shape_[i] << " vs " << y_shape_[i] << " in dimension " << i;
    }
    x_nums_ *= LongToSize(x_shape_[i]);
    y_nums_ *= LongToSize(y_shape_[i]);
    if (i < axis) {
      outer_size_ *= LongToSize(x_shape_[i]);
    } else if (i > axis) {
      inner_size_ *= LongToSize(x_shape_[i]);
    }
  }
  x_axis_size_ = LongToSize(x_shape_[axis]);
  y_axis_size_ = LongToSize(y_shape_[axis]);
}

template <typename T>
bool IndexAddCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexAddInputsNum, kernel_name_);
  auto *x = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *indices = reinterpret_cast<int32_t *>(inputs[kIndex1]->addr);
  auto *y = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  CheckParams();
  size_t x_axis_inner_size = x_axis_size_ * inner_size_;
  size_t y_axis_inner_size = y_axis_size_ * inner_size_;
  bool index_mismatch = 0;

  auto task = [this, &x, &y, &indices, &x_axis_inner_size, &y_axis_inner_size, &index_mismatch](const size_t start,
                                                                                                const size_t end) {
    for (size_t i = start; i < end; ++i) {
      // calc idx_y in y.shape[axis]
      const size_t y_axis_idx = (i / inner_size_) % y_axis_size_;
      // only process add operation when idx_x is valid
      if (indices[y_axis_idx] >= 0 && static_cast<size_t>(indices[y_axis_idx]) < x_axis_size_) {
        // calc idx_x in x.shape[axis]
        const size_t x_axis_idx = static_cast<size_t>(indices[y_axis_idx]);
        const size_t x_outer_idx = i / y_axis_inner_size;
        const size_t x_inner_idx = i % inner_size_;
        const size_t x_idx = x_outer_idx * x_axis_inner_size + x_axis_idx * inner_size_ + x_inner_idx;
        x[x_idx] += y[i];
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the indices out of range with input_shape: " << input_shapes_
                      << ".";
        index_mismatch = 1;
      }
    }
  };

  auto task_block = [this, &x, &y, &indices, &x_axis_inner_size, &y_axis_inner_size, &index_mismatch](
                      const size_t start, const size_t end) {
    for (size_t i = start; i < end; ++i) {
      const size_t y_outer_idx = i / y_axis_size_;
      const size_t y_axis_idx = i - y_outer_idx * y_axis_size_;
      if (indices[y_axis_idx] >= 0 && static_cast<size_t>(indices[y_axis_idx]) < x_axis_size_) {
        const size_t x_axis_idx = static_cast<size_t>(indices[y_axis_idx]);
        const size_t x_idx = y_outer_idx * x_axis_inner_size + x_axis_idx * inner_size_;
        const size_t y_idx = i * inner_size_;
        for (size_t j = 0; j < inner_size_; ++j) {
          x[x_idx + j] += y[y_idx + j];
        }
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the indices out of range with input_shape: " << input_shapes_
                      << ".";
        index_mismatch = 1;
      }
    }
  };

  auto heavy_task_block = [this, task_block](const size_t start, const size_t end) {
    task_block(start * y_axis_size_, end * y_axis_size_);
  };

  const float block_size = 1024;
  const size_t inner_block_size = 100;
  if (HasDuplicateIndex(indices, y_axis_size_)) {
    ParallelLaunch(heavy_task_block, outer_size_, SizeToFloat(CalcSizePerThread(outer_size_)), this);
  } else if (inner_size_ > 1 && inner_size_ <= inner_block_size) {
    ParallelLaunch(task_block, y_nums_ / inner_size_, block_size / inner_size_, this);
  } else {
    ParallelLaunch(task, y_nums_, block_size, this);
  }

  if (index_mismatch) {
    return false;
  }

  return true;
}

const std::vector<std::pair<KernelAttr, IndexAddCpuKernelMod::KernelRunFunc>> &IndexAddCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, IndexAddCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8)
       .AddOutInRef(0, 0),
     &IndexAddCpuKernelMod::LaunchKernel<uint8_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IndexAdd, IndexAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
