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

#include "plugin/device/gpu/kernel/sparse/sparse_segment_ops_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto Sparse_Segment_Sum = "SparseSegmentSum";
constexpr auto Sparse_Segment_Sum_With_Num_Segments = "SparseSegmentSumWithNumSegments";
constexpr auto Sparse_Segment_Sqrt_N = "SparseSegmentSqrtN";
constexpr auto Sparse_Segment_Sqrt_N_With_Num_Segments = "SparseSegmentSqrtNWithNumSegments";
constexpr size_t kNumber1 = 1;
constexpr size_t kNumber3 = 3;
constexpr size_t kNumber4 = 4;
}  // namespace

bool SparseSegmentOpsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  if (kernel_type_ == "SparseSegmentSum" || kernel_type_ == "SparseSegmentSqrtN") {
    flag_ = true;
  } else {
    flag_ = false;
  }
  size_t inputs_num = flag_ ? kNumber3 : kNumber4;
  size_t outputs_num = kNumber1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_type_)[index].second;
  unit_x_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  unit_idx_seg_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  return true;
}

int SparseSegmentOpsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  for (const auto &output : outputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  std::vector<int64_t> x_shape = inputs.at(kIndex0)->GetShapeVector();
  x_shape_0_ = x_shape[0];
  x_elements_ = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies{});
  outer_size_ = x_shape.front();
  inner_size_ = x_elements_ / x_shape.front();
  std::vector<int64_t> indices_shape = inputs.at(kIndex1)->GetShapeVector();
  idx_seg_elements_ = std::accumulate(indices_shape.begin(), indices_shape.end(), 1, std::multiplies{});
  output_dim0_ = LongToSize(output_shape.front());

  size_t input_x_size = x_elements_ * unit_x_size_;
  size_t input_idx_seg_size = idx_seg_elements_ * unit_idx_seg_size_;
  size_t output_size = output_elements_ * unit_x_size_;
  input_size_list_.push_back(input_x_size);
  input_size_list_.push_back(input_idx_seg_size);
  input_size_list_.push_back(input_idx_seg_size);
  if (flag_) {
    input_size_list_.push_back(unit_idx_seg_size_);
  }
  output_size_list_.push_back(output_size);
  workspace_size_list_.push_back((output_dim0_ + 1) * sizeof(size_t));
  return KRET_OK;
}

template <typename R, typename S>
bool SparseSegmentOpsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  R *x_ptr = GetDeviceAddress<R>(inputs, kIndex0);
  S *indices_ptr = GetDeviceAddress<S>(inputs, kIndex1);
  S *segment_ids_ptr = GetDeviceAddress<S>(inputs, kIndex2);
  R *y_ptr = GetDeviceAddress<R>(outputs, kIndex0);
  size_t *segment_pos_ptr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(x_ptr, indices_ptr, segment_ids_ptr, segment_pos_ptr, y_ptr)) {
    return false;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  std::vector<S> indices_host;
  std::vector<S> segment_ids_host;
  std::vector<S> num_segments_host;
  indices_host.resize(idx_seg_elements_);
  segment_ids_host.resize(idx_seg_elements_);
  num_segments_host.resize(kNumber1);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_host.data(), indices_ptr, idx_seg_elements_ * sizeof(S), cudaMemcpyDeviceToHost, stream),
    "cudaMemcpy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(segment_ids_host.data(), segment_ids_ptr,
                                                     idx_seg_elements_ * sizeof(S), cudaMemcpyDeviceToHost, stream),
                                     "cudaMemcpy failed.");
  if (!flag_) {
    auto num_segments_ptr = GetDeviceAddress<S>(inputs, kIndex3);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(num_segments_host.data(), num_segments_ptr, sizeof(S), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy failed.");
  }
  if (segment_ids_host[0] != 0 && flag_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', indices in 'segment_ids' should be contiguous and start from 0.";
  }
  for (size_t i = 1; i < idx_seg_elements_; i++) {
    if (segment_ids_host[i] < segment_ids_host[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids should be sorted.";
    }
    if (segment_ids_host[i] - segment_ids_host[i - 1] > 1 && flag_) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', indices in 'segment_ids' should be contiguous and start from 0.";
    }
  }
  if (segment_ids_host[idx_seg_elements_ - 1] >= num_segments_host[kIndex0] && !flag_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', num_segments must bigger than the last number of segment_ids.";
  }
  for (size_t i = 0; i < idx_seg_elements_; i++) {
    if (indices_host[i] >= static_cast<S>(x_shape_0_)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices out of range of x's first shape.";
    }
  }
  CalSparseSegmentCombination(kernel_type_, x_ptr, indices_ptr, segment_ids_ptr, segment_pos_ptr, outer_size_,
                              inner_size_, idx_seg_elements_, output_dim0_, y_ptr, device_id_, stream);
  return true;
}

std::map<std::string, std::vector<std::pair<KernelAttr, SparseSegmentOpsGpuKernelMod::SSLaunchFunc>>>
  SparseSegmentOpsGpuKernelMod::kernel_attr_map_ = {
    {Sparse_Segment_Sum,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeUInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeUInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeUInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeUInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int8_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int8_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int16_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int16_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int32_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int32_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int64_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int64_t>}}},
    {Sparse_Segment_Sum_With_Num_Segments,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeUInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeUInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeUInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeUInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int8_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt8),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int8_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int16_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int16_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int32_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int32_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int64_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int64_t>}}},
    {Sparse_Segment_Sqrt_N,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int64_t>}}},
    {Sparse_Segment_Sqrt_N_With_Num_Segments,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat16),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<half, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat32),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeFloat64),
       &SparseSegmentOpsGpuKernelMod::LaunchKernel<double, int64_t>}}}};  // kernel_attr_map_

std::vector<KernelAttr> SparseSegmentOpsGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_type_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'SparseSegmentOpsOp', only support these types: " << kernel::Map2Str(kernel_attr_map_)
                  << " currently, but got " << kernel_name_;
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSegmentOpsGpuKernelMod::SSLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseSegmentSum,
                                 []() { return std::make_shared<SparseSegmentOpsGpuKernelMod>(Sparse_Segment_Sum); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseSegmentSumWithNumSegments, []() {
  return std::make_shared<SparseSegmentOpsGpuKernelMod>(Sparse_Segment_Sum_With_Num_Segments);
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseSegmentSqrtN, []() {
  return std::make_shared<SparseSegmentOpsGpuKernelMod>(Sparse_Segment_Sqrt_N);
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseSegmentSqrtNWithNumSegments, []() {
  return std::make_shared<SparseSegmentOpsGpuKernelMod>(Sparse_Segment_Sqrt_N_With_Num_Segments);
});
}  // namespace kernel
}  // namespace mindspore
