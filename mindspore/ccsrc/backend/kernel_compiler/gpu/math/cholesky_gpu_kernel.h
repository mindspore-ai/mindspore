/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CHOLESKY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CHOLESKY_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "backend/kernel_compiler/gpu/cuda_impl/triangle_matrix_copy_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kCholeskyInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kCholeskyOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kCholeskyDefaultShape = 1;
constexpr size_t kCholeskyNormalShape = 2;
constexpr size_t kCholeskyBatchedShape = 3;

template <typename T>
class ScipyCholeskyGpuKernel : public GpuKernel {
 public:
  ScipyCholeskyGpuKernel() = default;
  ~ScipyCholeskyGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    // here all addresses are malloc by cuda, so deal with them as device's address.
    auto input1_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto output_addr = GetDeviceAddress<T>(outputs, kDim0);

    auto d_array_addr = GetDeviceAddress<T *>(workspace, kDim0);
    auto d_info_array_addr = GetDeviceAddress<int>(workspace, kDim1);

    for (size_t i = 0; i < batch_; i++) {
      h_array_[i] = input1_addr + i * lda_ * m_;
    }

    // 5. copy input's addr to d_array_addr
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(d_array_addr, h_array_.data(), sizeof(T *) * batch_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");

    // 6. solve to cholesky factorization according to cuSolver api, outputs have been written to input's matrix.
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT(
        kernel_node_, cusolverDnSpotrfBatched(handle_, uplo_, m_, d_array_addr, lda_, d_info_array_addr, batch_),
        "cusolver cholesky batched Fail");
    } else if constexpr (std::is_same_v<T, double>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT(
        kernel_node_, cusolverDnDpotrfBatched(handle_, uplo_, m_, d_array_addr, lda_, d_info_array_addr, batch_),
        "cusolver cholesky batched Fail");
    } else {
      MS_LOG(EXCEPTION) << "cholesky factorization do not support other data type but only float or double, right now.";
    }
    size_t output_elements = outputs.at(kDim0)->size / unit_size_;
    // 7. copy results from written input's matrix to output's matrix by up or lower flag.
    ScipyTriangleMatrixCopy(input1_addr, output_addr, uplo_, output_elements, ldb_, m_,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    lower_ = static_cast<bool>(GetAttr<bool>(kernel_node, "lower"));
    if (lower_) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }
    // 1. get CuSolver Dense matrix handler
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    // 2. get Cublas handler
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

    auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

    // 3. check input shape not null
    bool is_null_input = CHECK_NULL_INPUT(in_shape);
    if (is_null_input) {
      MS_LOG(EXCEPTION) << "For 'PureCholeskyGpuKernel', input is null";
    }
    // 4. calculate input size
    if (!InitInputSize(in_shape)) {
      MS_LOG(EXCEPTION) << "For 'PureCholeskyGpuKernel', input shape init failed.";
    }
    return true;
  }

 private:
  bool InitInputSize(const std::vector<size_t> &in_shape) {
    if (in_shape.size() == kCholeskyDefaultShape) {
      batch_ = 1;
      cho_row_ = in_shape.at(kDim0);
      cho_col_ = cho_row_;
    } else if (in_shape.size() == kCholeskyNormalShape) {
      batch_ = 1;
      cho_row_ = in_shape.at(kDim0);
      cho_col_ = in_shape.at(kDim1);
    } else if (in_shape.size() == kCholeskyBatchedShape) {
      batch_ = SizeToInt(in_shape.at(kDim0));
      cho_row_ = in_shape.at(kDim1);
      cho_col_ = in_shape.at(kDim2);
    } else {
      MS_LOG(ERROR) << "Input Only support Rank 2 OR 3";
      return false;
    }
    if (cho_row_ != cho_col_) {
      MS_LOG(ERROR) << "Cholesky need square matrix as input.";
      return false;
    }
    // set matrix row or col to be lead dimension
    m_ = SizeToInt(cho_row_);
    lda_ = m_;
    ldb_ = m_;
    h_array_.resize(batch_);
    InitSizeLists();
    return true;
  }

  void InitSizeLists() override {
    input_size_ = batch_ * m_ * lda_ * unit_size_;
    input_size_list_.push_back(input_size_);

    output_size_ = batch_ * m_ * lda_ * unit_size_;
    output_size_list_.push_back(output_size_);

    size_t workspace_size = batch_ * sizeof(T *);
    workspace_size_list_.resize(kDim2);
    workspace_size_list_[kDim0] = workspace_size;

    workspace_size = batch_ * sizeof(int);
    workspace_size_list_[kDim1] = workspace_size;
  }

  size_t unit_size_{sizeof(T)};
  size_t cho_row_{0};
  size_t cho_col_{0};
  size_t batch_{0};
  size_t m_{0};
  size_t lda_{0};
  size_t ldb_{0};
  size_t input_size_{0};
  size_t output_size_{0};
  cusolverDnHandle_t handle_{nullptr};
  cublasHandle_t blas_handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  bool lower_{false};
  std::vector<T *> h_array_{};
  std::vector<size_t> input_size_list_{};
  std::vector<size_t> output_size_list_{};
  std::vector<size_t> workspace_size_list_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
