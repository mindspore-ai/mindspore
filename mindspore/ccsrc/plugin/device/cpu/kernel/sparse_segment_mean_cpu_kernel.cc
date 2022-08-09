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

#include "plugin/device/cpu/kernel/sparse_segment_mean_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool SparseSegmentMeanCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t inputs_num = 3;
  constexpr size_t outputs_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseSegmentMeanCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto indices_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  auto y_shape = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  auto batch_rank = base_operator->get_batch_rank();
  batch_size_ = std::accumulate(x_shape.begin(), x_shape.begin() + batch_rank, size_t(1), std::multiplies{});
  outer_size_ = x_shape.at(LongToSize(batch_rank));
  inner_size_ = std::accumulate(x_shape.begin() + batch_rank + 1, x_shape.end(), size_t(1), std::multiplies{});
  x_size_ = inner_size_ * outer_size_;
  indices_size_ = indices_shape.at(LongToSize(batch_rank));
  y_size_ = std::accumulate(y_shape.begin() + batch_rank, y_shape.end(), size_t(1), std::multiplies{});
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSegmentMeanCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ptr = reinterpret_cast<DataType *>(inputs[kIndex0]->addr);
  auto indices_ptr = reinterpret_cast<IndexType *>(inputs[kIndex1]->addr);
  auto segment_ids_ptr = reinterpret_cast<IndexType *>(inputs[kIndex2]->addr);
  auto y_ptr = reinterpret_cast<DataType *>(outputs[kIndex0]->addr);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(x_ptr, indices_ptr, segment_ids_ptr, y_ptr)) {
    return false;
  }
  // Check 'indices' validity.
  for (size_t ii = 0; ii < batch_size_; ii++) {
    auto offset = ii * indices_size_;
    for (size_t k = 0; k < indices_size_; k++) {
      auto i = k + offset;
      if (indices_ptr[i] < 0 || indices_ptr[i] >= static_cast<IndexType>(outer_size_)) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << i << "-th 'indices'[" << indices_ptr[i]
                      << "] is out of range[0, " << outer_size_ << ").";
        return false;
      }
    }
  }
  // Check 'segment_ids' validity.
  for (size_t ii = 0; ii < batch_size_; ii++) {
    auto offset = ii * indices_size_;
    if (segment_ids_ptr[offset] < 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'segment_ids' must be non-negative.";
      return false;
    }
    for (size_t k = 1; k < indices_size_; k++) {
      auto i = k + offset;
      if (segment_ids_ptr[i] < segment_ids_ptr[i - 1]) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'segment_ids' must be sorted.";
        return false;
      }
    }
  }
  auto task = [this, &x_ptr, &indices_ptr, &segment_ids_ptr, &y_ptr](size_t start, size_t end) {
    for (size_t ii = 0; ii < batch_size_; ii++) {
      size_t x_batch_offset = ii * x_size_;
      size_t index_batch_offset = ii * indices_size_;
      size_t y_batch_offset = ii * y_size_;
      size_t segment_length = 1;
      IndexType pre_segment_id = -1;
      for (size_t k = 0; k < indices_size_; ++k) {
        auto i = index_batch_offset + k;
        auto x_offset = x_batch_offset + LongToSize(indices_ptr[i]) * inner_size_;
        auto y_offset = y_batch_offset + LongToSize(segment_ids_ptr[i]) * inner_size_;
        // Reset the empty segments by setting output[i] = 0.
        for (auto sid = pre_segment_id + 1; sid <= segment_ids_ptr[i]; sid++) {
          auto reset_y_ptr = y_ptr + (y_batch_offset + LongToSize(sid) * inner_size_);
          std::fill(reset_y_ptr + start, reset_y_ptr + end, DataType(0));
        }
        pre_segment_id = segment_ids_ptr[i];
        // Accumulate the specific segment.
        for (size_t j = start; j < end; ++j) {
          y_ptr[y_offset + j] = x_ptr[x_offset + j] + y_ptr[y_offset + j];
        }
        // Since reduce type is mean, divide by length of segment.
        if (i + 1 == indices_size_ || segment_ids_ptr[i] != segment_ids_ptr[i + 1]) {
          for (size_t j = start; j < end; ++j) {
            y_ptr[y_offset + j] /= static_cast<DataType>(segment_length);
          }
          segment_length = 1;
        } else {
          segment_length++;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, inner_size_, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseSegmentMeanCpuKernelMod::SparseSegmentMeanLaunchFunc>>
  SparseSegmentMeanCpuKernelMod::func_list_ = {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSegmentMeanCpuKernelMod::LaunchKernel<double, int64_t>},
  }};

std::vector<KernelAttr> SparseSegmentMeanCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSegmentMeanCpuKernelMod::SparseSegmentMeanLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentMean, SparseSegmentMeanCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
