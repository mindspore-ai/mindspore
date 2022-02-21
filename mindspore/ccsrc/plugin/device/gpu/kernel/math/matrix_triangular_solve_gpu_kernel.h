/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kAVectorxDimNum = 1;
constexpr auto kAMatrixDimNum = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
constexpr size_t kShape3D = 3;
constexpr size_t kIndexAArray = 0;
constexpr size_t kIndexDstArray = 1;
constexpr size_t kIndexBBuffer = 2;
constexpr size_t kIndexBTransposeShape = 3;
constexpr size_t kIndexBTransposeAxis = 4;
template <typename T>
class MatrixTriangularSolveGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixTriangularSolveGpuKernelMod() = default;
  ~MatrixTriangularSolveGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    auto inputa_addr = GetDeviceAddress<T>(inputs, 0);
    auto inputb_addr = GetDeviceAddress<T>(inputs, 1);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);

    // if b is not a vector, solve b in the workspace
    T *dst = nullptr;
    if (n_ == 1) {
      dst = output_addr;
    } else {
      dst = GetDeviceAddress<T>(workspace, kIndexBBuffer);
    }

    const size_t batched_b_size = batch_ * m_ * n_;
    if (n_ == 1) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dst, inputb_addr, batched_b_size * sizeof(T), cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync dst failed");
    } else {
      // No matter how many batch dimensions the batched matrix b has, use their cumulative multiplication batch.
      // In order to convert row major matrix b(batch, m, n) to col major matrix b'(batch, m, n),
      // the following operation is equivalent to:
      // b' = b.tarnspose(batch, n, m).reshape(batch, m, n)
      size_t host_transpose_b_shape[kShape3D] = {batch_, m_, n_};
      size_t host_transpose_b_axis[kShape3D] = {kDim0, kDim2, kDim1};
      auto dev_transpose_b_shape = GetDeviceAddress<size_t>(workspace, kIndexBTransposeShape);
      auto dev_transpose_b_axis = GetDeviceAddress<size_t>(workspace, kIndexBTransposeAxis);
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_transpose_b_shape, host_transpose_b_shape, kShape3D * sizeof(T *),
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "memcpy input a axis workspace failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_transpose_b_axis, host_transpose_b_axis, kShape3D * sizeof(T *),
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "memcpy input b axis workspace failed");
      CalTranspose(batched_b_size, inputb_addr, dev_transpose_b_shape, dev_transpose_b_axis, kShape3D, dst,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    // index calculation
    auto device_a_array_addr = GetDeviceAddress<T *>(workspace, kIndexAArray);
    auto device_dst_array_addr = GetDeviceAddress<T *>(workspace, kIndexDstArray);
    for (size_t i = 0; i < batch_; i++) {
      host_a_array_[i] = inputa_addr + i * m_ * m_;
      host_dst_array_[i] = dst + i * m_ * n_;
    }

    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(device_a_array_addr, host_a_array_.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(device_dst_array_addr, host_dst_array_.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");

    T alpha = 1;
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasStrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_, &alpha,
                           device_a_array_addr, lda_, device_dst_array_addr, ldb_, batch_),
        "cublas trsm Fail");
    } else {
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasDtrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_, &alpha,
                           device_a_array_addr, lda_, device_dst_array_addr, ldb_, batch_),
        "cublas trsm Fail");
    }

    // if x is not a vector, do transpose
    if (n_ != 1) {
      // in order to convert col major matrix x'(m x n) to row major matrix x'(m x n),
      // the following operation is equivalent to:
      // x = x'.reshape(n, m).T
      size_t host_transpose_b_shape[kShape3D] = {batch_, n_, m_};
      size_t host_transpose_b_axis[kShape3D] = {kDim0, kDim2, kDim1};
      auto dev_transpose_b_shape = GetDeviceAddress<size_t>(workspace, kIndexBTransposeShape);
      auto dev_transpose_b_axis = GetDeviceAddress<size_t>(workspace, kIndexBTransposeAxis);
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_transpose_b_shape, host_transpose_b_shape, kShape3D * sizeof(T *),
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "memcpy input a axis workspace failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_transpose_b_axis, host_transpose_b_axis, kShape3D * sizeof(T *),
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "memcpy input b axis workspace failed");
      CalTranspose(batched_b_size, dst, dev_transpose_b_shape, dev_transpose_b_axis, kShape3D, output_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

    InitShape(kernel_node);
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    lda_ = SizeToInt(m_);
    ldb_ = SizeToInt(m_);

    if (AnfAlgo::HasNodeAttr("adjoint", kernel_node)) {
      // MatrixTriangularSolve attribute
      bool trans = AnfAlgo::GetNodeAttr<bool>(kernel_node, "adjoint");
      // converting row major to col major is the same as reverting the trans flag
      trans_ = trans ? CUBLAS_OP_N : CUBLAS_OP_T;
      if (AnfAlgo::HasNodeAttr("trans", kernel_node)) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the attribute 'adjoint' and 'trans' could not exist at the same time.";
      }
    } else {
      bool lower = AnfAlgo::GetNodeAttr<bool>(kernel_node, "lower");
      // reverting the trans flag by default, so also flip the lower flag
      lower = !lower;
      uplo_ = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      bool unit_diagonal = AnfAlgo::GetNodeAttr<bool>(kernel_node, "unit_diagonal");
      unit_diagonal_ = unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
      const std::string trans = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "trans");
      // converting row major to col major is the same as reverting the trans flag
      if (trans == "N") {
        trans_ = CUBLAS_OP_T;
      } else if (trans == "T") {
        trans_ = CUBLAS_OP_N;
      } else if (trans == "C") {
        trans_ = CUBLAS_OP_N;
      } else {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', trans should be in [N, T, C], but got [" << trans << "].";
      }
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t unit_size = sizeof(T);
    size_t a_size = batch_ * m_ * m_ * unit_size;
    size_t b_size = batch_ * m_ * n_ * unit_size;
    input_size_list_ = {a_size, b_size};
    output_size_list_ = {b_size};
    if (n_ != 1) {
      workspace_size_list_ = {
        // workspace for batched a
        batch_ * sizeof(T *),
        // workspace for batched b
        batch_ * sizeof(T *),
        // workspace for transposed b
        b_size,
        // workspace for b transpose shape
        kShape3D * sizeof(size_t *),
        // workspace for b transpose axis
        kShape3D * sizeof(size_t *),
      };
    } else {
      workspace_size_list_ = {
        // workspace for batched a
        batch_ * sizeof(T *),
        // workspace for batched b
        batch_ * sizeof(T *),
      };
    }
  }

  void InitShape(const CNodePtr &kernel_node) {
    auto a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

    is_null_input_ =
      CHECK_SHAPE_NULL(a_shape, kernel_name_, "input_a") || CHECK_SHAPE_NULL(b_shape, kernel_name_, "input_b");
    // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
    size_t a_dims = a_shape.size();
    size_t aRowIndex = a_dims - kRowIndex;
    m_ = a_shape[aRowIndex];
    size_t b_sims = b_shape.size();
    bool vector_b = b_sims == a_dims - 1;
    if (vector_b) {
      n_ = 1;
    } else {
      n_ = b_shape[b_sims - 1];
    }
    batch_ = 1;
    for (size_t batch = 0; batch < a_dims - kRowIndex; ++batch) {
      batch_ *= a_shape[batch];
    }
    host_a_array_.resize(batch_);
    host_dst_array_.resize(batch_);
  }

 private:
  size_t m_{0};
  size_t n_{0};
  size_t batch_{1};
  int lda_{0};
  int ldb_{0};
  bool is_null_input_{false};
  std::vector<T *> host_a_array_;
  std::vector<T *> host_dst_array_;
  cublasHandle_t blas_handle_{nullptr};
  cublasFillMode_t uplo_{CUBLAS_FILL_MODE_UPPER};
  cublasOperation_t trans_{CUBLAS_OP_N};
  cublasDiagType_t unit_diagonal_{CUBLAS_DIAG_NON_UNIT};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
