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

#include "plugin/device/gpu/kernel/math/sparse_segment_mean_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool SparseSegmentMeanGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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

int SparseSegmentMeanGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  auto x_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
  outer_size_ = LongToSize(x_shape.front());
  inner_size_ = LongToSize(x_size / x_shape.front());
  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  auto indices_size = std::accumulate(indices_shape.begin(), indices_shape.end(), 1, std::multiplies<int64_t>());
  indices_size_ = LongToSize(indices_size);
  auto y_shape = outputs.at(kIndex0)->GetShapeVector();
  segment_size_ = LongToSize(y_shape.front());
  workspace_size_list_.push_back((segment_size_ + 1) * sizeof(size_t));
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSegmentMeanGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto segment_ids_ptr = GetDeviceAddress<IndexType>(inputs, kIndex2);
  auto segment_pos_ptr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);

  bool is_nullptr = (x_ptr == nullptr) || (indices_ptr == nullptr) || (segment_ids_ptr == nullptr) ||
                    (segment_pos_ptr == nullptr) || (y_ptr == nullptr);
  if (is_nullptr) {
    return false;
  }

  SparseSegmentMean(x_ptr, indices_ptr, segment_ids_ptr, segment_pos_ptr, y_ptr, outer_size_, inner_size_,
                    indices_size_, segment_size_, cuda_stream);
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
