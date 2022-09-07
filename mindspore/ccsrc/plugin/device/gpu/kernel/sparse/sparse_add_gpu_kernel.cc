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

#include "plugin/device/gpu/kernel/sparse/sparse_add_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_add_impl.cuh"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kSparseAddInputsNum = 7;
constexpr int kSparseAddOutputsNum = 3;
constexpr size_t kSparseAddIndex0 = 0;
constexpr size_t kSparseAddIndex1 = 1;
constexpr size_t kSparseAddIndex2 = 2;
constexpr size_t kSparseAddIndex3 = 3;
constexpr size_t kSparseAddIndex4 = 4;
constexpr size_t kSparseAddIndex5 = 5;
constexpr size_t kSparseAddIndex6 = 6;
constexpr size_t kSparseAddIndex7 = 7;
constexpr size_t kSparseAddIndex8 = 8;
}  // namespace

bool SparseAddGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseAddOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  is_need_retrieve_output_shape_ = true;  // SparseAdd is a dynamic shape operator.
  kernel_func_ = func_list_[index].second;
  indices_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kSparseAddIndex0).first);
  values_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kSparseAddIndex1).first);
  threshold_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kSparseAddIndex2).first);
  return true;
}

void SparseAddGpuKernelMod::ResetResource() noexcept {
  real_output_size_ = 0;
  a_indices_shape_.clear();
  a_values_shape_.clear();
  dense_shape_.clear();
  b_indices_shape_.clear();
  b_values_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void SparseAddGpuKernelMod::CalWorkSpace() {
  auto a_values_num = a_values_shape_[0];
  auto b_values_num = b_values_shape_[0];
  size_t a_value_index_size = a_values_num * sizeof(size_t);
  size_t b_value_index_size = b_values_num * sizeof(size_t);
  size_t is_from_a_size = (a_values_num + b_values_num) * sizeof(bool);
  size_t whole_values_size = (a_values_num + b_values_num) * values_size_;
  size_t place_holder_size = ((a_values_num > b_values_num) ? a_values_num : b_values_num) * sizeof(size_t);
  size_t indices_size = (a_values_num + b_values_num) * sizeof(int64_t);
  size_t threshold_valid_size = (a_values_num + b_values_num) * sizeof(bool);
  size_t res_store_mem_size = ((a_values_num > b_values_num) ? a_values_num : b_values_num) * values_size_;
  size_t sum_count_size = sizeof(int64_t);
  workspace_size_list_.push_back(a_value_index_size);
  workspace_size_list_.push_back(b_value_index_size);
  workspace_size_list_.push_back(is_from_a_size);
  workspace_size_list_.push_back(whole_values_size);
  workspace_size_list_.push_back(place_holder_size);
  workspace_size_list_.push_back(indices_size);
  workspace_size_list_.push_back(threshold_valid_size);
  workspace_size_list_.push_back(res_store_mem_size);
  workspace_size_list_.push_back(sum_count_size);
}

int SparseAddGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  outputs_ = outputs;
  auto a_indices_shape = inputs.at(kSparseAddIndex0)->GetShapeVector();
  auto a_values_shape = inputs.at(kSparseAddIndex1)->GetShapeVector();
  auto dense_shape = inputs.at(kSparseAddIndex2)->GetShapeVector();
  auto b_indices_shape = inputs.at(kSparseAddIndex3)->GetShapeVector();
  auto b_values_shape = inputs.at(kSparseAddIndex4)->GetShapeVector();
  rank_ = a_indices_shape.at(1);
  (void)std::transform(a_indices_shape.begin(), a_indices_shape.end(), std::back_inserter(a_indices_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  (void)std::transform(a_values_shape.begin(), a_values_shape.end(), std::back_inserter(a_values_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  (void)std::transform(dense_shape.begin(), dense_shape.end(), std::back_inserter(dense_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  (void)std::transform(b_indices_shape.begin(), b_indices_shape.end(), std::back_inserter(b_indices_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  (void)std::transform(b_values_shape.begin(), b_values_shape.end(), std::back_inserter(b_values_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  // rank_ = input_shape_.size();
  a_indices_size_ = std::accumulate(a_indices_shape_.begin(), a_indices_shape_.end(), 1, std::multiplies{});
  a_values_size_ = std::accumulate(a_values_shape_.begin(), a_values_shape_.end(), 1, std::multiplies{});
  dense_shape_size_ = std::accumulate(dense_shape_.begin(), dense_shape_.end(), 1, std::multiplies{});
  b_indices_size_ = std::accumulate(b_indices_shape_.begin(), b_indices_shape_.end(), 1, std::multiplies{});
  b_values_size_ = std::accumulate(b_values_shape_.begin(), b_values_shape_.end(), 1, std::multiplies{});
  if (a_indices_size_ == 0 || a_values_size_ == 0 || dense_shape_size_ == 0 || b_indices_size_ == 0 ||
      b_values_size_ == 0) {
    return KRET_UNKNOWN_SHAPE;
  }

  auto a_row_num = a_values_shape_[0];
  auto b_row_num = b_values_shape_[0];
  input_size_list_.push_back(a_indices_size_ * indices_size_);
  input_size_list_.push_back(a_values_size_ * values_size_);
  input_size_list_.push_back(b_indices_size_ * indices_size_);
  input_size_list_.push_back(b_values_size_ * values_size_);
  CalWorkSpace();
  output_size_list_.push_back((a_row_num + b_row_num) * rank_ * indices_size_);
  output_size_list_.push_back((a_row_num + b_row_num) * values_size_);
  output_size_list_.push_back(dense_shape_size_ * indices_size_);
  return KRET_OK;
}

template <typename T, typename S, typename K>
bool SparseAddGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (a_indices_size_ == 0 || a_values_size_ == 0 || dense_shape_size_ == 0 || b_indices_size_ == 0 ||
      b_values_size_ == 0) {
    return true;
  }

  auto a_indices_ptr = GetDeviceAddress<T>(inputs, kSparseAddIndex0);
  MS_EXCEPTION_IF_NULL(a_indices_ptr);
  auto a_values_ptr = GetDeviceAddress<S>(inputs, kSparseAddIndex1);
  MS_EXCEPTION_IF_NULL(a_values_ptr);
  auto dense_shape_ptr = GetDeviceAddress<T>(inputs, kSparseAddIndex2);
  MS_EXCEPTION_IF_NULL(dense_shape_ptr);
  auto b_indices_ptr = GetDeviceAddress<T>(inputs, kSparseAddIndex3);
  MS_EXCEPTION_IF_NULL(b_indices_ptr);
  auto b_values_ptr = GetDeviceAddress<S>(inputs, kSparseAddIndex4);
  MS_EXCEPTION_IF_NULL(b_values_ptr);
  auto threshold_ptr = GetDeviceAddress<K>(inputs, kSparseAddIndex6);
  MS_EXCEPTION_IF_NULL(threshold_ptr);
  auto sum_indices_ptr = GetDeviceAddress<T>(outputs, kSparseAddIndex0);
  MS_EXCEPTION_IF_NULL(sum_indices_ptr);
  auto sum_values_ptr = GetDeviceAddress<S>(outputs, kSparseAddIndex1);
  MS_EXCEPTION_IF_NULL(sum_values_ptr);
  auto sum_shape_ptr = GetDeviceAddress<T>(outputs, kSparseAddIndex2);
  MS_EXCEPTION_IF_NULL(sum_indices_ptr);

  auto a_value_index_ptr = GetDeviceAddress<size_t>(workspace, kSparseAddIndex0);
  MS_EXCEPTION_IF_NULL(a_value_index_ptr);
  auto b_value_index_ptr = GetDeviceAddress<size_t>(workspace, kSparseAddIndex1);
  MS_EXCEPTION_IF_NULL(b_value_index_ptr);
  auto is_from_a_ptr = GetDeviceAddress<bool>(workspace, kSparseAddIndex2);
  auto whole_values_ptr = GetDeviceAddress<S>(workspace, kSparseAddIndex3);
  auto place_holder_index_ptr = GetDeviceAddress<size_t>(workspace, kSparseAddIndex4);
  auto indices_ptr = GetDeviceAddress<int64_t>(workspace, kSparseAddIndex5);
  auto threshold_valid_ptr = GetDeviceAddress<bool>(workspace, kSparseAddIndex6);
  auto res_store_mem_ptr = GetDeviceAddress<S>(workspace, kSparseAddIndex7);
  auto sum_count_ptr = GetDeviceAddress<int64_t>(workspace, kSparseAddIndex8);

  auto a_indices_num = a_values_shape_[0];
  auto b_indices_num = b_values_shape_[0];

  // workspace/output mem reset
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(a_value_index_ptr, static_cast<size_t>(0), workspace.at(kSparseAddIndex0)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(b_value_index_ptr, static_cast<size_t>(0), workspace.at(kSparseAddIndex1)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(is_from_a_ptr, static_cast<bool>(0), workspace.at(kSparseAddIndex2)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(whole_values_ptr, 0, workspace.at(kSparseAddIndex3)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(place_holder_index_ptr, static_cast<size_t>(0), workspace.at(kSparseAddIndex4)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(indices_ptr, static_cast<int64_t>(0), workspace.at(kSparseAddIndex5)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(threshold_valid_ptr, static_cast<bool>(0), workspace.at(kSparseAddIndex6)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(res_store_mem_ptr, 0, workspace.at(kSparseAddIndex7)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(sum_count_ptr, static_cast<int64_t>(0), workspace.at(kSparseAddIndex8)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(sum_indices_ptr, static_cast<T>(0), outputs.at(kSparseAddIndex0)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(sum_values_ptr, static_cast<T>(0), outputs.at(kSparseAddIndex1)->size, cuda_stream_),
    "For SparseAdd, failed to cudaMemset.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For SparseAdd, cudaStreamSynchronize failed.");

  SparseAdd(a_indices_ptr, a_values_ptr, b_indices_ptr, b_values_ptr, sum_indices_ptr, sum_values_ptr,
            a_value_index_ptr, b_value_index_ptr, is_from_a_ptr, whole_values_ptr, place_holder_index_ptr, indices_ptr,
            threshold_valid_ptr, a_indices_num, b_indices_num, res_store_mem_ptr, sum_count_ptr, threshold_ptr,
            device_id_, cuda_stream_);
  // Get dynamic shape
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_size_, sum_count_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream_),
    "For SparseAdd, cudaMemcpyAsync failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(sum_shape_ptr, dense_shape_ptr, dense_shape_size_ * indices_size_,
                                                     cudaMemcpyDeviceToDevice, cuda_stream_),
                                     "For SparseAdd, cudaMemcpyAsync failed.");
  return true;
}

#define GPU_SPARSE_ADD_KERNEL_REGISTER(ms_index_type, ms_value_type, ms_thr_type, index_type, value_type, thr_type) \
  {                                                                                                                 \
    KernelAttr()                                                                                                    \
      .AddInputAttr(ms_index_type)                                                                                  \
      .AddInputAttr(ms_value_type)                                                                                  \
      .AddInputAttr(ms_index_type)                                                                                  \
      .AddInputAttr(ms_index_type)                                                                                  \
      .AddInputAttr(ms_value_type)                                                                                  \
      .AddInputAttr(ms_index_type)                                                                                  \
      .AddInputAttr(ms_thr_type)                                                                                    \
      .AddOutputAttr(ms_index_type)                                                                                 \
      .AddOutputAttr(ms_value_type)                                                                                 \
      .AddOutputAttr(ms_index_type),                                                                                \
      &SparseAddGpuKernelMod::LaunchKernel<index_type, value_type, thr_type>                                        \
  }

void SparseAddGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For SparseAdd cudaStreamSynchronized failed.");
  std::vector<int64_t> sum_indices_shape = {real_output_size_, static_cast<int32_t>(rank_)};
  std::vector<int64_t> sum_values_shape = {real_output_size_};
  std::vector<int64_t> dense_shape(dense_shape_.begin(), dense_shape_.end());
  outputs_[kSparseAddIndex0]->SetShapeVector(sum_indices_shape);
  outputs_[kSparseAddIndex1]->SetShapeVector(sum_values_shape);
  outputs_[kSparseAddIndex2]->SetShapeVector(dense_shape);
}

std::vector<std::pair<KernelAttr, SparseAddGpuKernelMod::SparseAddLaunchFunc>> SparseAddGpuKernelMod::func_list_ = {
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int64_t, int8_t, int8_t),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int64_t, int16_t, int16_t),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t, int32_t, int32_t),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t, int64_t),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32, int64_t, float, float),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64, int64_t, double, double),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex64, kNumberTypeFloat32, int64_t, cuComplex, float),
  GPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex128, kNumberTypeFloat64, int64_t, cuDoubleComplex,
                                 double),
};

std::vector<KernelAttr> SparseAddGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseAddGpuKernelMod::SparseAddLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseAdd, SparseAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
