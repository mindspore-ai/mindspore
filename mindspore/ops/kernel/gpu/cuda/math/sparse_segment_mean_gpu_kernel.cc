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

#include "kernel/gpu/cuda/math/sparse_segment_mean_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool SparseSegmentMeanGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  constexpr size_t inputs_num = 3;
  constexpr size_t outputs_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseSegmentMeanGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  auto y_shape = outputs.at(kIndex0)->GetShapeVector();
  batch_rank_ =
    primitive_->HasAttr("batch_rank") ? LongToSize(GetValue<int64_t>(primitive_->GetAttr("batch_rank"))) : 0;
  batch_size_ = std::accumulate(x_shape.begin(), x_shape.begin() + batch_rank_, size_t(1), std::multiplies{});
  outer_size_ = LongToSize(x_shape.at(batch_rank_));
  inner_size_ = std::accumulate(x_shape.begin() + batch_rank_ + 1, x_shape.end(), size_t(1), std::multiplies{});
  x_size_ = inner_size_ * outer_size_;
  indices_size_ = LongToSize(indices_shape.at(batch_rank_));
  y_size_ = std::accumulate(y_shape.begin() + batch_rank_, y_shape.end(), size_t(1), std::multiplies{});
  segment_size_ = LongToSize(y_shape.at(batch_rank_));
  // In vmap case, all batch of the 'segment_ids' are same, therefore, workspace just one copy.
  workspace_size_list_.push_back((segment_size_ + 1) * sizeof(size_t));
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSegmentMeanGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto segment_ids_ptr = GetDeviceAddress<IndexType>(inputs, kIndex2);
  auto segment_pos_ptr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(x_ptr, indices_ptr, segment_ids_ptr, segment_pos_ptr, y_ptr)) {
    return false;
  }

  auto status = SparseSegmentMean(x_ptr, indices_ptr, segment_ids_ptr, segment_pos_ptr, y_ptr, outer_size_, inner_size_,
                                  indices_size_, segment_size_, x_size_, y_size_, batch_size_, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseSegmentMeanGpuKernelMod::SparseSegmentMeanLaunchFunc>>
  SparseSegmentMeanGpuKernelMod::func_list_ = {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSegmentMeanGpuKernelMod::LaunchKernel<double, int64_t>},
  }};

std::vector<KernelAttr> SparseSegmentMeanGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSegmentMeanGpuKernelMod::SparseSegmentMeanLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSegmentMean, SparseSegmentMeanGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
