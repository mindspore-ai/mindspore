/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TRSM_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TRSM_SOLVE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr auto kAVectorxDimNum = 1;
constexpr auto kAMatrixDimNum = 2;
template <typename T>
class TrsmGpuKernel : public GpuKernel {
 public:
  TrsmGpuKernel() = default;
  ~TrsmGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    auto inputA_addr = GetDeviceAddress<T>(inputs, 0);
    auto inputb_addr = GetDeviceAddress<T>(inputs, 1);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);

    // if b is not a vector, solve b in the workspace
    T *dst = nullptr;
    if (n_ == 1) {
      dst = output_addr;
    } else {
      dst = GetDeviceAddress<T>(workspace, 0);
    }

    if (n_ == 1) {
      const size_t batch = m_ * n_;
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dst, inputb_addr, batch * sizeof(T), cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync dst failed");
    } else {
      T alpha = 1;
      T beta = 0;
      // in order to convert row major matrix b(m x n) to col major matrix b'(m x n),
      // the following operation is equivalent to:
      // b' = b.T.reshape(m, n)
      if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                     cublasSgeam(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_T, m_, n_, &alpha, inputb_addr,
                                                 n_, &beta, inputb_addr, n_, dst, m_),
                                     "cublas transpose b Fail");
      } else {
        CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                     cublasDgeam(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_T, m_, n_, &alpha, inputb_addr,
                                                 n_, &beta, inputb_addr, n_, dst, m_),
                                     "cublas transpose b Fail");
      }
    }

    T alpha = 1;
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasStrsm(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_,
                                               &alpha, inputA_addr, lda_, dst, ldb_),
                                   "cublas trsm Fail");
    } else {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasDtrsm(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_,
                                               &alpha, inputA_addr, lda_, dst, ldb_),
                                   "cublas trsm Fail");
    }

    // if x is not a vector, do transpose
    if (n_ != 1) {
      T alpha = 1;
      T beta = 0;
      // in order to convert col major matrix x'(m x n) to row major matrix x'(m x n),
      // the following operation is equivalent to:
      // x = x'.reshape(n, m).T
      if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS_RET_WITH_EXCEPT(
          kernel_node_,
          cublasSgeam(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_T, n_, m_, &alpha, dst, m_, &beta, dst, m_, output_addr, n_),
          "cublas transpose x Fail");
      } else {
        CHECK_CUBLAS_RET_WITH_EXCEPT(
          kernel_node_,
          cublasDgeam(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_T, n_, m_, &alpha, dst, m_, &beta, dst, m_, output_addr, n_),
          "cublas transpose x Fail");
      }
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(A_shape) || CHECK_NULL_INPUT(b_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'TrsmGpuKernel', input is null";
      InitSizeLists();
      return true;
    }

    if (A_shape[kDim0] != A_shape[kDim1]) {
      MS_LOG(EXCEPTION) << "wrong array shape, A should be a squre matrix, but got [" << A_shape[kDim0] << " X "
                        << A_shape[kDim1] << "]";
    }
    m_ = A_shape[kDim0];

    if (b_shape.size() != kAVectorxDimNum && b_shape.size() != kAMatrixDimNum) {
      MS_LOG(EXCEPTION) << "wrong array shape, b should be 1D or 2D, but got [" << b_shape.size() << "] dimensions";
    }
    if (b_shape[kDim0] != m_) {
      MS_LOG(EXCEPTION) << "wrong array shape, b should match the shape of A, excepted [" << m_ << "] but got ["
                        << b_shape[kDim0] << "]";
    }
    if (b_shape.size() == kAVectorxDimNum || (b_shape.size() == kAMatrixDimNum && b_shape[kDim1] == 1)) {
      n_ = 1;
    } else {
      n_ = b_shape[kDim1];
    }

    lda_ = SizeToInt(m_);
    ldb_ = SizeToInt(m_);

    const std::string trans = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "trans");
    // converting row major to col major is the same as reverting the trans flag
    if (trans == "N") {
      trans_ = CUBLAS_OP_T;
    } else if (trans == "T") {
      trans_ = CUBLAS_OP_N;
    } else {
      MS_LOG(EXCEPTION) << "trans should be in [N, T], but got [" << trans << "]";
    }

    bool lower = AnfAlgo::GetNodeAttr<bool>(kernel_node, "lower");
    // reverting the trans flag by default, so also flip the lower flag
    lower = !lower;
    if (lower) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }

    bool unit_diagonal = AnfAlgo::GetNodeAttr<bool>(kernel_node, "unit_diagonal");
    if (unit_diagonal) {
      unit_diagonal_ = CUBLAS_DIAG_UNIT;
    } else {
      unit_diagonal_ = CUBLAS_DIAG_NON_UNIT;
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t unit_size = sizeof(T);
    size_t A_size = m_ * m_ * unit_size;
    size_t b_size = m_ * n_ * unit_size;
    input_size_list_ = {A_size, b_size};
    output_size_list_ = {b_size};
    if (n_ != 1) {
      workspace_size_list_ = {b_size};
    }
  }

 private:
  size_t m_{0};
  size_t n_{0};
  int lda_{0};
  int ldb_{0};
  bool is_null_input_{false};
  cublasHandle_t blas_handle_{nullptr};
  cublasFillMode_t uplo_{CUBLAS_FILL_MODE_UPPER};
  cublasOperation_t trans_{CUBLAS_OP_N};
  cublasDiagType_t unit_diagonal_{CUBLAS_DIAG_NON_UNIT};
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TRSM_SOLVE_GPU_KERNEL_H_
