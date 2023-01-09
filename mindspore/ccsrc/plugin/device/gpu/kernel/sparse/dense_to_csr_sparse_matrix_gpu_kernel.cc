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

#include "plugin/device/gpu/kernel/sparse/dense_to_csr_sparse_matrix_gpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, DenseToCSRSparseMatrixKernelMod::LaunchFunc>>
  DenseToCSRSparseMatrixKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<bool, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<int8_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<int16_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<int, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<int64_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<uint8_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<uint16_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<uint, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<uint64_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<half, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<cuComplex, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &DenseToCSRSparseMatrixKernelMod::LaunchKernel<cuDoubleComplex, int>}};

std::vector<KernelAttr> DenseToCSRSparseMatrixKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, DenseToCSRSparseMatrixKernelMod::LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T, typename S>
bool DenseToCSRSparseMatrixKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  S *dev_nd_strides_ = GetDeviceAddress<S>(workspace, kIndex0);
  S *dev_nd_indices_ = GetDeviceAddress<S>(workspace, kIndex1);
  S *dev_row_indices_ = GetDeviceAddress<S>(workspace, kIndex2);
  S *dense_shape_addr = GetDeviceAddress<S>(outputs, kIndex0);
  S *batch_pointers_addr = GetDeviceAddress<S>(outputs, kIndex1);
  S *row_pointers_addr = GetDeviceAddress<S>(outputs, kIndex2);
  S *col_indices_addr = GetDeviceAddress<S>(outputs, kIndex3);
  T *values_addr = GetDeviceAddress<T>(outputs, kIndex4);

  std::vector<S> nd_strides_;
  std::vector<S> nd_indices_;
  nd_strides_.resize(dim_indices_last_, 0);
  nd_indices_.resize(dim_indices_last_, 0);

  if (dim_indices_last_ > 0) {
    nd_strides_[dim_indices_last_ - 1] = input_shapes_[dim_indices_last_ - 1];
    nd_indices_[dim_indices_last_ - 1] = dims_[kIndex1];
  }
  for (size_t i = dim_indices_last_ - 1; i > 0; --i) {
    nd_strides_[i - 1] = input_shapes_[i - 1];
    nd_indices_[i - 1] = nd_indices_[i] * input_shapes_[i];
  }

  const size_t strides_len = sizeof(S) * nd_strides_.size();
  const size_t indices_len = sizeof(S) * nd_indices_.size();
  std::vector<S> input_shapes_host(input_shapes_.begin(), input_shapes_.end());

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(dev_nd_strides_, &nd_strides_[kIndex0], strides_len, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync failed in DenseToCSRSparseMatrixKernelMod::Launch.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(dev_nd_indices_, &nd_indices_[kIndex0], indices_len, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync failed in DenseToCSRSparseMatrixKernelMod::Launch.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(dense_shape_addr, &input_shapes_host[kIndex0], indices_len, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync failed in DenseToCSRSparseMatrixKernelMod::Launch.");

  size_t num_batches = (is_batch_csr_) ? input_shapes_[kIndex0] : 1;
  // row pointers need to be set to zero to avoid any blank rows.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(row_pointers_addr, 0, outputs[kIndex2]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemset failed in DenseToCSRSparseMatrixKernelMod::Launch.");

  int *device_flag = GetDeviceAddress<int>(workspace, 4);
  if (!is_batch_csr_) {
    std::vector<S> batch_ptr_host{};
    batch_ptr_host.emplace_back(0);
    batch_ptr_host.emplace_back(nnz_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(batch_pointers_addr, &batch_ptr_host[kIndex0], sizeof(S) * (num_batches + 1),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in DenseToCSRSparseMatrixKernelMod::Launch.");
    auto ret = GatherNd(input_addr, indices_addr, values_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
                        dev_nd_strides_, dev_nd_indices_, device_flag, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (ret.first >= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the operator gathernd's indices[" << ret.first
                        << "]: " << ret.second << ", does not index into input_shape: " << input_shapes_ << ".";
    }
    CallSplitIndices2D(indices_addr, dev_row_indices_, col_indices_addr, nnz_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    cusparseXcoo2csr(handle_, dev_row_indices_, nnz_, m_, row_pointers_addr, CUSPARSE_INDEX_BASE_ZERO);
  } else {
    S *dev_batch_indices_ = GetDeviceAddress<S>(workspace, 3);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(batch_pointers_addr, 0, outputs[kIndex1]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemset failed in DenseToCSRSparseMatrixKernelMod::Launch.");

    auto ret = GatherNd(input_addr, indices_addr, values_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
                        dev_nd_strides_, dev_nd_indices_, device_flag, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (ret.first >= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the operator gathernd's indices[" << ret.first
                        << "]: " << ret.second << ", does not index into input_shape: " << input_shapes_ << ".";
    }
    CallSplitIndices3D(indices_addr, dev_batch_indices_, dev_row_indices_, col_indices_addr, nnz_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    CallNNZPerBatch(dev_batch_indices_, batch_pointers_addr, nnz_, num_batches + 1,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    std::vector<S> host_batch_pointers(batch_pointers_shapes_[kIndex0], 0);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(host_batch_pointers.data(), batch_pointers_addr, sizeof(S) * (num_batches + 1),
                      cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed for device batch pointers to host.");
    auto row_num = input_shapes_[kIndex1];
    for (size_t i = 0; i < host_batch_pointers.size() - 1; ++i) {
      S *temp_row_indices_addr = dev_row_indices_ + host_batch_pointers[i];
      S *temp_row_pointers_addr = row_pointers_addr + i * (row_num + 1);
      int temp_nnz = host_batch_pointers[i + 1] - host_batch_pointers[i];
      if (temp_nnz != 0) {
        cusparseXcoo2csr(handle_, temp_row_indices_addr, temp_nnz, row_num, temp_row_pointers_addr,
                         CUSPARSE_INDEX_BASE_ZERO);
      }
    }
  }
  return true;
}

int DenseToCSRSparseMatrixKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shapes_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetDeviceShapeAdaptively());
  indices_shapes_ = Convert2SizeTClipNeg(inputs[kIndex1]->GetDeviceShapeAdaptively());
  dense_shape_shapes_ = Convert2SizeTClipNeg(outputs[kIndex0]->GetDeviceShapeAdaptively());
  batch_pointers_shapes_ = Convert2SizeTClipNeg(outputs[kIndex1]->GetDeviceShapeAdaptively());
  row_pointers_shapes_ = Convert2SizeTClipNeg(outputs[kIndex2]->GetDeviceShapeAdaptively());
  col_indices_shapes_ = Convert2SizeTClipNeg(outputs[kIndex3]->GetDeviceShapeAdaptively());
  value_shapes_ = Convert2SizeTClipNeg(outputs[kIndex4]->GetDeviceShapeAdaptively());

  nnz_ = value_shapes_[kIndex0];
  m_ = input_shapes_[kIndex0];

  Reshape();

  dim_indices_last_ = dims_[dims_.size() - 1];

  rank_ = input_shapes_.size();
  constexpr size_t kBatchCSR = 3;
  is_batch_csr_ = (rank_ == kBatchCSR) ? true : false;

  workspace_size_list_.push_back(sizeof(output_size_list_.at(kIndex0)) * dim_indices_last_);
  workspace_size_list_.push_back(sizeof(output_size_list_.at(kIndex0)) * dim_indices_last_);
  workspace_size_list_.push_back(sizeof(output_size_list_.at(kIndex0)) * nnz_);
  workspace_size_list_.push_back(sizeof(output_size_list_.at(kIndex0)) * nnz_);
  workspace_size_list_.push_back(sizeof(int));
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DenseToCSRSparseMatrix, DenseToCSRSparseMatrixKernelMod);
}  // namespace kernel
}  // namespace mindspore
