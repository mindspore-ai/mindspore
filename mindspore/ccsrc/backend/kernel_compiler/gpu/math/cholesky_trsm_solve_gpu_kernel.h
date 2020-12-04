/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CHOLESKY_TRSM_SOLVE_GPU_KERNEL_H
#define MINDSPORE_CHOLESKY_TRSM_SOLVE_GPU_KERNEL_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/identity_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/matrix_split_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class CholeskyTrsmGpuKernel : public GpuKernel {
 public:
  CholeskyTrsmGpuKernel() : batch_(0), m_(0), lda_(0), is_null_input_(false), handle_(nullptr) {}
  ~CholeskyTrsmGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    if (!use_split_matrix) {
      LaunchNonSplitMatrix(inputs, workspace, outputs, stream_ptr);
    } else {
      LaunchSplitMatrix(inputs, workspace, outputs, stream_ptr);
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    split_dim = static_cast<int>(GetAttr<int64_t>(kernel_node, "split_dim"));
    if (split_dim == 0) {
      InitDim0(kernel_node, in_shape);
    } else {
      if (in_shape.size() != 2) {
        MS_LOG(ERROR) << "CholeskyTrsm Split Matrix Need Input Rank as 2.";
      }
      if (in_shape[0] != in_shape[1]) {
        MS_LOG(ERROR) << "CholeskyTrsm Split Matrix Need Square Matrix as Input.";
      }
      InitDimOthers(kernel_node, in_shape);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    if (!use_split_matrix) {
      size_t unit_size = sizeof(T);
      size_t input_size = batch_ * m_ * lda_ * unit_size;
      input_size_list_.push_back(input_size);
      size_t output_size = batch_ * m_ * lda_ * unit_size;
      output_size_list_.push_back(output_size);
      size_t workspace_size = batch_ * sizeof(T *);
      workspace_size_list_.push_back(workspace_size);
      workspace_size = batch_ * sizeof(T *);
      workspace_size_list_.push_back(workspace_size);
      workspace_size = batch_ * sizeof(int);
      workspace_size_list_.push_back(workspace_size);
    } else {
      size_t unit_size = sizeof(T);
      size_t input_size = height * width * unit_size;
      input_size_list_.push_back(input_size);
      size_t output_size = batch_ * m_ * lda_ * unit_size;
      output_size_list_.push_back(output_size);
      size_t workspace_size = batch_ * sizeof(T *);
      workspace_size_list_.push_back(workspace_size);
      workspace_size = batch_ * sizeof(T *);
      workspace_size_list_.push_back(workspace_size);
      workspace_size = batch_ * sizeof(int);
      workspace_size_list_.push_back(workspace_size);
      workspace_size = batch_ * m_ * lda_ * unit_size;
      workspace_size_list_.push_back(workspace_size);
    }
  }

 private:
  void LaunchNonSplitMatrix(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    auto input1_addr = GetDeviceAddress<T>(inputs, 0);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);
    auto d_array_addr = GetDeviceAddress<T *>(workspace, 0);
    auto d_identity_addr = GetDeviceAddress<T *>(workspace, 1);
    auto d_info_array_addr = GetDeviceAddress<int>(workspace, 2);
    for (size_t i = 0; i < batch_; i++) {
      h_array[i] = input1_addr + i * lda_ * m_;
      h_identity[i] = output_addr + i * ldb_ * m_;
      CHECK_CUDA_RET_WITH_ERROR(
        kernel_node_,
        cudaMemcpyAsync(output_addr + i * ldb_ * m_, h_identity_data.data(), sizeof(T) * ldb_ * m_,
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cuda memcopy Fail");
    }
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(d_array_addr, h_array.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(d_identity_addr, h_identity.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUSOLVER_RET_WITH_EXCEPT(
      kernel_node_, cusolverDnSpotrfBatched(handle_, uplo, m_, d_array_addr, lda_, d_info_array_addr, batch_),
      "cusolver cholesky batched Fail");
    float alpha = 1;
    CHECK_CUBLAS_RET_WITH_EXCEPT(
      kernel_node_,
      cublasStrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_, m_, &alpha,
                         d_array_addr, lda_, d_identity_addr, ldb_, batch_),
      "cublas trsm batched Fail");
  }
  void LaunchSplitMatrix(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    auto input1_addr = GetDeviceAddress<T>(inputs, 0);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);
    auto d_array_addr = GetDeviceAddress<T *>(workspace, 0);
    auto d_identity_addr = GetDeviceAddress<T *>(workspace, 1);
    auto d_info_array_addr = GetDeviceAddress<int>(workspace, 2);
    auto d_batch_input_addr = GetDeviceAddress<T>(workspace, 3);
    for (size_t i = 0; i < batch_; i++) {
      h_array[i] = d_batch_input_addr + i * lda_ * m_;
      h_identity[i] = output_addr + i * ldb_ * m_;
    }
    Identity(batch_ * split_dim * split_dim, split_dim, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    MatrixSplit(batch_ * split_dim * split_dim, split_dim, width, input1_addr, d_batch_input_addr,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(d_array_addr, h_array.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(d_identity_addr, h_identity.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUSOLVER_RET_WITH_EXCEPT(
      kernel_node_, cusolverDnSpotrfBatched(handle_, uplo, m_, d_array_addr, lda_, d_info_array_addr, batch_),
      "cusolver cholesky batched Fail");
    float alpha = 1;
    CHECK_CUBLAS_RET_WITH_EXCEPT(
      kernel_node_,
      cublasStrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_, m_, &alpha,
                         d_array_addr, lda_, d_identity_addr, ldb_, batch_),
      "cublas trsm batched Fail");
  }
  void InitDim0(const CNodePtr &kernel_node, const std::vector<size_t> &in_shape) {
    use_split_matrix = false;
    if (in_shape.size() == 2) {
      batch_ = 1;
      if (in_shape[0] != in_shape[1]) {
        MS_LOG(ERROR) << "CholeskyTrsm need square matrix as input.";
      }
    } else if (in_shape.size() == 3) {
      batch_ = SizeToInt(in_shape[0]);
      if (in_shape[1] != in_shape[2]) {
        MS_LOG(ERROR) << "CholeskyTrsm need square matrix as input.";
      }
    } else {
      MS_LOG(ERROR) << "Input Only support Rank 2 OR 3";
    }

    m_ = SizeToInt(in_shape[1]);
    lda_ = m_;
    ldb_ = m_;
    h_array.resize(batch_);
    h_identity.resize(batch_);
    h_identity_data.resize(m_ * m_);
    for (size_t i = 0; i < m_; i++) {
      for (size_t j = 0; j < m_; j++) {
        if (i == j) {
          h_identity_data[i * m_ + j] = 1;
        } else {
          h_identity_data[i * m_ + j] = 0;
        }
      }
    }
  }
  void InitDimOthers(const CNodePtr &kernel_node, const std::vector<size_t> &in_shape) {
    height = in_shape[0];
    width = in_shape[1];
    if (SizeToInt(height) <= split_dim) {
      use_split_matrix = false;
      batch_ = 1;
      m_ = SizeToInt(in_shape[1]);
      lda_ = m_;
      ldb_ = m_;
      h_array.resize(batch_);
      h_identity.resize(batch_);
      h_identity_data.resize(m_ * m_);
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < m_; j++) {
          if (i == j) {
            h_identity_data[i * m_ + j] = 1;
          } else {
            h_identity_data[i * m_ + j] = 0;
          }
        }
      }
    } else {
      use_split_matrix = true;
      int batch = SizeToInt(in_shape[1]) / split_dim;
      res_dim = in_shape[1] - batch * split_dim;
      if (res_dim == 0) {
        batch_ = batch;
      } else {
        batch_ = batch + 1;
      }
      m_ = split_dim;
      lda_ = m_;
      ldb_ = m_;
      h_array.resize(batch_);
      h_identity.resize(batch_);
      h_identity_data.resize(m_ * m_);
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < m_; j++) {
          if (i == j) {
            h_identity_data[i * m_ + j] = 1;
          } else {
            h_identity_data[i * m_ + j] = 0;
          }
        }
      }
    }
  }
  size_t batch_;
  size_t m_;
  size_t lda_;
  size_t ldb_;
  int res_dim;
  int split_dim;
  bool is_null_input_;
  bool use_split_matrix;
  size_t height;
  size_t width;
  cusolverDnHandle_t handle_;
  cublasHandle_t blas_handle_;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  std::vector<T *> h_array;
  std::vector<T *> h_identity;
  std::vector<T> h_identity_data;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
