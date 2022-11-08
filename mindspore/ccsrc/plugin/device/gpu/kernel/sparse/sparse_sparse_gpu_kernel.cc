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

#include "plugin/device/gpu/kernel/sparse/sparse_sparse_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto Sparse_Sparse_Maximum = "SparseSparseMaximum";
constexpr auto Sparse_Sparse_Minimum = "SparseSparseMinimum";
constexpr int kSparseSparseInputsNum = 6;
constexpr int kSparseSparseOutputsNum = 2;
constexpr size_t kSparseSparseIndex0 = 0;
constexpr size_t kSparseSparseIndex1 = 1;
constexpr size_t kSparseSparseIndex2 = 2;
constexpr size_t kSparseSparseIndex3 = 3;
constexpr size_t kSparseSparseIndex4 = 4;
constexpr size_t kSparseSparseIndex5 = 5;
}  // namespace

bool SparseSparseGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSparseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSparseOutputsNum, kernel_name_);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  is_need_retrieve_output_shape_ = true;
  indices_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kSparseSparseIndex0).first);
  values_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kSparseSparseIndex1).first);
  return true;
}

int SparseSparseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  outputs_ = outputs;
  auto a_indices_shape = inputs.at(kSparseSparseIndex0)->GetShapeVector();
  auto a_values_shape = inputs.at(kSparseSparseIndex1)->GetShapeVector();
  auto dense_shape = inputs.at(kSparseSparseIndex2)->GetShapeVector();
  auto b_indices_shape = inputs.at(kSparseSparseIndex3)->GetShapeVector();
  auto b_values_shape = inputs.at(kSparseSparseIndex4)->GetShapeVector();
  rank_ = a_indices_shape.at(1);
  auto a_indices_size = std::accumulate(a_indices_shape.begin(), a_indices_shape.end(), 1, std::multiplies<int64_t>());
  auto a_values_size = std::accumulate(a_values_shape.begin(), a_values_shape.end(), 1, std::multiplies<int64_t>());
  auto dense_shape_size = std::accumulate(dense_shape.begin(), dense_shape.end(), 1, std::multiplies<int64_t>());
  auto b_indices_size = std::accumulate(b_indices_shape.begin(), b_indices_shape.end(), 1, std::multiplies<int64_t>());
  auto b_values_size = std::accumulate(b_values_shape.begin(), b_values_shape.end(), 1, std::multiplies<int64_t>());
  if (a_indices_size == 0 || a_values_size == 0 || dense_shape_size == 0 || b_indices_size == 0 || b_values_size == 0) {
    is_null_input_ = true;
  }

  auto a_row_num = a_values_shape[0];
  auto b_row_num = b_values_shape[0];
  a_indices_num_ = a_values_shape[0];
  b_indices_num_ = b_values_shape[0];
  auto a_values_num = a_values_shape[0];
  auto b_values_num = b_values_shape[0];
  size_t ab_status = (a_values_num * b_values_num) * sizeof(int64_t);
  size_t sum = sizeof(int64_t);
  size_t ab_status1 = (a_values_num + b_values_num) * sizeof(int64_t);
  size_t ab_status2 = (a_values_num + b_values_num) * sizeof(int64_t);

  input_size_list_.push_back(a_indices_size * indices_size_);
  input_size_list_.push_back(a_values_size * values_size_);
  input_size_list_.push_back(b_indices_size * indices_size_);
  input_size_list_.push_back(b_values_size * values_size_);
  output_size_list_.push_back((a_row_num + b_row_num) * rank_ * indices_size_);
  output_size_list_.push_back((a_row_num + b_row_num) * values_size_);
  workspace_size_list_.push_back(ab_status);
  workspace_size_list_.push_back(sum);
  workspace_size_list_.push_back(ab_status1);
  workspace_size_list_.push_back(ab_status2);
  return KRET_OK;
}

template <typename T, typename S>
bool SparseSparseGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto a_indices_ptr = GetDeviceAddress<T>(inputs, kSparseSparseIndex0);
  MS_EXCEPTION_IF_NULL(a_indices_ptr);
  auto a_values_ptr = GetDeviceAddress<S>(inputs, kSparseSparseIndex1);
  MS_EXCEPTION_IF_NULL(a_values_ptr);
  auto dense_shape_ptr1 = GetDeviceAddress<T>(inputs, kSparseSparseIndex2);
  MS_EXCEPTION_IF_NULL(dense_shape_ptr1);
  auto b_indices_ptr = GetDeviceAddress<T>(inputs, kSparseSparseIndex3);
  MS_EXCEPTION_IF_NULL(b_indices_ptr);
  auto b_values_ptr = GetDeviceAddress<S>(inputs, kSparseSparseIndex4);
  MS_EXCEPTION_IF_NULL(b_values_ptr);
  auto dense_shape_ptr2 = GetDeviceAddress<T>(inputs, kSparseSparseIndex5);
  MS_EXCEPTION_IF_NULL(dense_shape_ptr2);
  auto sum_indices_ptr = GetDeviceAddress<T>(outputs, kSparseSparseIndex0);
  MS_EXCEPTION_IF_NULL(sum_indices_ptr);
  auto sum_values_ptr = GetDeviceAddress<S>(outputs, kSparseSparseIndex1);
  MS_EXCEPTION_IF_NULL(sum_values_ptr);
  auto ab_status_ptr = GetDeviceAddress<int64_t>(workspace, kSparseSparseIndex0);
  MS_EXCEPTION_IF_NULL(ab_status_ptr);
  auto sum_ptr = GetDeviceAddress<int64_t>(workspace, kSparseSparseIndex1);
  MS_EXCEPTION_IF_NULL(sum_ptr);
  auto ab_status_ptr1 = GetDeviceAddress<int64_t>(workspace, kSparseSparseIndex2);
  MS_EXCEPTION_IF_NULL(ab_status_ptr1);
  auto ab_status_ptr2 = GetDeviceAddress<int64_t>(workspace, kSparseSparseIndex3);
  MS_EXCEPTION_IF_NULL(ab_status_ptr2);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(sum_ptr, static_cast<int64_t>(1), workspace.at(kSparseSparseIndex1)->size, cuda_stream_),
    "For SparseSparseOperators, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(ab_status_ptr1, static_cast<int64_t>(kSparseSparseIndex3),
                                                    workspace.at(kSparseSparseIndex2)->size, cuda_stream_),
                                    "For SparseSparseOperators, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For SparseSparseOperators, cudaStreamSynchronize failed.");
  std::vector<int64_t> x1_shape(rank_);
  std::vector<int64_t> x2_shape(rank_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&x1_shape[0], dense_shape_ptr1, rank_ * sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream_),
    "For SparseSparseOperators, cudaMemcpyAsync failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&x2_shape[0], dense_shape_ptr2, rank_ * sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream_),
    "For SparseSparseOperators, cudaMemcpyAsync failed.");
  for (int64_t n = 0; n < rank_; n++) {
    if (x1_shape[n] != x2_shape[n]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', operands' shapes do not match.";
    }
  }
  if (kernel_name_ == "SparseSparseMaximum") {
    SparseSparseMaximum(a_indices_ptr, a_values_ptr, b_indices_ptr, b_values_ptr, sum_indices_ptr, sum_values_ptr,
                        ab_status_ptr, sum_ptr, a_indices_num_, b_indices_num_, rank_, cuda_stream_, device_id_,
                        ab_status_ptr1, ab_status_ptr2);
  } else {
    SparseSparseMinimum(a_indices_ptr, a_values_ptr, b_indices_ptr, b_values_ptr, sum_indices_ptr, sum_values_ptr,
                        ab_status_ptr, sum_ptr, a_indices_num_, b_indices_num_, rank_, cuda_stream_, device_id_,
                        ab_status_ptr1, ab_status_ptr2);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_size_, sum_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream_),
    "For SparseSparseOperators, failed to cudaMemset.");
  return true;
}

void SparseSparseGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For SparseSparseOperators, cudaStreamSynchronize failed.");
  std::vector<int64_t> sum_indices_shape = {real_output_size_, static_cast<int64_t>(rank_)};
  std::vector<int64_t> sum_values_shape = {real_output_size_};
  outputs_[kSparseSparseIndex0]->SetShapeVector(sum_indices_shape);
  outputs_[kSparseSparseIndex1]->SetShapeVector(sum_values_shape);
}

std::vector<std::pair<KernelAttr, SparseSparseGpuKernelMod::SparseSparseFunc>> SparseSparseGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SparseSparseGpuKernelMod::LaunchKernel<int64_t, double>},
};

std::vector<KernelAttr> SparseSparseGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseSparseFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSparseMaximum, SparseSparseGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSparseMinimum, SparseSparseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
