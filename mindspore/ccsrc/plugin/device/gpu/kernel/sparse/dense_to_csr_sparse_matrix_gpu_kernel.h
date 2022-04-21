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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_DENSE_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_DENSE_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dense_to_csr_sparse_matrix_gpu_kernel.cuh"

namespace mindspore {
namespace kernel {
#define HOST_TO_DEVICE(host_ptr, device_ptr, size)                                                                     \
  {                                                                                                                    \
    CHECK_CUDA_RET_WITH_EXCEPT(                                                                                        \
      kernel_node_,                                                                                                    \
      cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)), \
      "cudaMemcpyAsync failed in DenseToCSRSparseMatrixKernelMod::Launch.");                                           \
  }

#define DEVICE_TO_HOST(device_ptr, host_ptr, size)                                                                     \
  {                                                                                                                    \
    CHECK_CUDA_RET_WITH_ERROR(                                                                                         \
      kernel_node_,                                                                                                    \
      cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)), \
      "cudaMemcpyAsync failed for device batch pointers to host.");                                                    \
  }

#define DEVICE_MEMSET(device_ptr, value, size)                                                                       \
  {                                                                                                                  \
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,                                                                         \
                               cudaMemsetAsync(device_ptr, value, size, reinterpret_cast<cudaStream_t>(stream_ptr)), \
                               "cudaMemset failed in DenseToCSRSparseMatrixKernelMod::Launch.");                     \
  }

template <typename T, typename S>
class DenseToCSRSparseMatrixKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  DenseToCSRSparseMatrixKernelMod() {
    ResetResource();
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle();
  }
  ~DenseToCSRSparseMatrixKernelMod() {}

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
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

    if (!memcpy_flag_) {
      const size_t strides_len = sizeof(S) * nd_strides_.size();
      const size_t indices_len = sizeof(S) * nd_indices_.size();
      std::vector<S> input_shapes_host(input_shapes_.begin(), input_shapes_.end());
      HOST_TO_DEVICE(&nd_strides_[kIndex0], dev_nd_strides_, strides_len);
      HOST_TO_DEVICE(&nd_indices_[kIndex0], dev_nd_indices_, indices_len);
      HOST_TO_DEVICE(&input_shapes_host[kIndex0], dense_shape_addr, indices_len);
      memcpy_flag_ = true;
    }

    size_t num_batches = (is_batch_csr_) ? input_shapes_[kIndex0] : 1;
    // row pointers need to be set to zero to avoid any blank rows.
    DEVICE_MEMSET(row_pointers_addr, 0, sizeof(S) * outputs[kIndex2]->size);

    if (!is_batch_csr_) {
      std::vector<S> batch_ptr_host{};
      batch_ptr_host.emplace_back(0);
      batch_ptr_host.emplace_back(nnz_);
      HOST_TO_DEVICE(&batch_ptr_host[kIndex0], batch_pointers_addr, sizeof(S) * (num_batches + 1));
      GatherNd(input_addr, indices_addr, values_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], dev_nd_strides_,
               dev_nd_indices_, reinterpret_cast<cudaStream_t>(stream_ptr));
      CallSplitIndices2D(indices_addr, dev_row_indices_, col_indices_addr, nnz_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
      cusparseXcoo2csr(handle_, dev_row_indices_, nnz_, m_, row_pointers_addr, CUSPARSE_INDEX_BASE_ZERO);
    } else {
      S *dev_batch_indices_ = GetDeviceAddress<S>(workspace, 3);
      DEVICE_MEMSET(batch_pointers_addr, 0, sizeof(S) * outputs[kIndex1]->size);
      GatherNd(input_addr, indices_addr, values_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], dev_nd_strides_,
               dev_nd_indices_, reinterpret_cast<cudaStream_t>(stream_ptr));
      CallSplitIndices3D(indices_addr, dev_batch_indices_, dev_row_indices_, col_indices_addr, nnz_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
      CallNNZPerBatch(dev_batch_indices_, batch_pointers_addr, nnz_, reinterpret_cast<cudaStream_t>(stream_ptr));
      std::vector<S> host_batch_pointers(batch_pointers_shapes_[kIndex0], 0);
      DEVICE_TO_HOST(batch_pointers_addr, host_batch_pointers.data(), sizeof(S) * (num_batches + 1))
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

  bool Init(const CNodePtr &kernel_node) override {
    constexpr size_t kBatchCSR = 3;
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    memcpy_flag_ = false;

    input_shapes_ = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, kIndex0);
    indices_shapes_ = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, kIndex1);
    dense_shape_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, kIndex0);
    batch_pointers_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, kIndex1);
    row_pointers_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, kIndex2);
    col_indices_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, kIndex3);
    value_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, kIndex4);

    nnz_ = value_shapes_[kIndex0];
    m_ = input_shapes_[kIndex0];

    Reshape();

    size_t dim_indices_last = dims_[dims_.size() - 1];
    nd_strides_.resize(dim_indices_last, 0);
    nd_indices_.resize(dim_indices_last, 0);

    if (dim_indices_last > 0) {
      nd_strides_[dim_indices_last - 1] = input_shapes_[dim_indices_last - 1];
      nd_indices_[dim_indices_last - 1] = dims_[kIndex1];
    }
    for (size_t i = dim_indices_last - 1; i > 0; --i) {
      nd_strides_[i - 1] = input_shapes_[i - 1];
      nd_indices_[i - 1] = nd_indices_[i] * input_shapes_[i];
    }

    rank_ = input_shapes_.size();
    is_batch_csr_ = (rank_ == kBatchCSR) ? true : false;
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_list_.clear();
    workspace_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(input_shapes_));
    input_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(indices_shapes_));
    workspace_size_list_.push_back(sizeof(S) * nd_strides_.size());
    workspace_size_list_.push_back(sizeof(S) * nd_indices_.size());
    workspace_size_list_.push_back(sizeof(S) * nnz_);
    workspace_size_list_.push_back(sizeof(S) * batch_pointers_shapes_[kIndex0]);
    output_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(dense_shape_shapes_));
    output_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(batch_pointers_shapes_));
    output_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(row_pointers_shapes_));
    output_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(col_indices_shapes_));
    output_size_list_.push_back(common::AnfAlgo::TensorSizeInByte<T>(value_shapes_));
  }

 private:
  cusparseHandle_t handle_{nullptr};
  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> dense_shape_shapes_;
  std::vector<size_t> batch_pointers_shapes_;
  std::vector<size_t> row_pointers_shapes_;
  std::vector<size_t> col_indices_shapes_;
  std::vector<size_t> value_shapes_;
  std::vector<size_t> dims_;
  std::vector<S> nd_strides_;
  std::vector<S> nd_indices_;

  bool is_batch_csr_{false};
  bool memcpy_flag_{false};
  int nnz_;
  int m_;
  int rank_;

  void Reshape() {
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); i++) {
      dim_of_indices *= indices_shapes_[i];
    }

    size_t dim_after_indices = 1;
    size_t dim_indices_last = indices_shapes_[indices_shapes_.size() - IntToSize(1)];
    for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
      dim_after_indices *= input_shapes_[i];
    }
    dims_.emplace_back(dim_of_indices);
    dims_.emplace_back(dim_after_indices);
    dims_.emplace_back(dim_indices_last);
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
