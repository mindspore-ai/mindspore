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
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/convert_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/triangle_matrix_copy_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kLuInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kLuOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kLuDefaultShape = 1;
constexpr size_t kLuNormalShape = 2;

template <typename T>
class LUGpuKernel : public GpuKernel {
 public:
  LUGpuKernel() = default;
  ~LUGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "cusolverDnSetStream failed");
    auto input_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto output_addr = GetDeviceAddress<T>(outputs, kDim0);
    int *piv_output_addr = nullptr;
    if (pivot_on_) {
      piv_output_addr = GetDeviceAddress<int>(outputs, kDim1);
    }

    auto info_output_addr = GetDeviceAddress<int>(outputs, kDim2);

    // 4. query working space of getrf
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT(kernel_node_,
                                     cusolverDnSgetrf_bufferSize(handle_, m_, m_, input_addr, lda_, &lwork_),
                                     "cusolver query lu work size fail");

      if (cudaMalloc(reinterpret_cast<void **>(&d_work_), unit_size_ * lwork_) != cudaSuccess) {
        MS_LOG(EXCEPTION) << "cusolver malloc work size fail";
      }

      CHECK_CUSOLVER_RET_WITH_EXCEPT(
        kernel_node_, cusolverDnSgetrf(handle_, m_, m_, input_addr, lda_, d_work_, piv_output_addr, info_output_addr),
        "cusolver lu fail");

    } else if constexpr (std::is_same_v<T, double>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT(kernel_node_,
                                     cusolverDnDgetrf_bufferSize(handle_, m_, m_, input_addr, lda_, &lwork_),
                                     "cusolver query lu work size fail");
      // 5. malloc device working space of getrf

      if (cudaMalloc(reinterpret_cast<void **>(&d_work_), unit_size_ * lwork_) != cudaSuccess) {
        MS_LOG(EXCEPTION) << "cusolver malloc work size fail";
      }

      // 6. solve to lu factorization according to cuSolver api, outputs have been written to input's matrix.
      CHECK_CUSOLVER_RET_WITH_EXCEPT(
        kernel_node_, cusolverDnDgetrf(handle_, m_, m_, input_addr, lda_, d_work_, piv_output_addr, info_output_addr),
        "cusolver lu fail");
    } else {
      MS_LOG(EXCEPTION) << "cholesky factorization do not support other data type but only float or double, right now.";
    }
    // 7. copy results from written input's matrix to output's matrix.
    //    if (cudaMemcpy(output_addr, input_addr, lda_ * m_ * unit_size_, cudaMemcpyDeviceToDevice) != cudaSuccess) {
    //      MS_LOG(EXCEPTION) << "memcpy lu output fail.";
    //    }
    MatrixCopy(input_addr, output_addr, lda_ * m_, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (d_work_) {
      cudaFree(d_work_);
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    // 1. get CuSolver Dense matrix handler
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    // 2. check input shape not null
    bool is_null_input = CHECK_NULL_INPUT(in_shape);
    if (is_null_input) {
      MS_LOG(EXCEPTION) << "For 'PureCholeskyGpuKernel', input is null";
    }
    // 3. calculate input size
    if (!InitInputSize(in_shape)) {
      MS_LOG(EXCEPTION) << "For 'PureCholeskyGpuKernel', input shape init failed.";
    }
    return true;
  }

 private:
  bool InitInputSize(const std::vector<size_t> &in_shape) {
    if (in_shape.size() == kLuDefaultShape) {
      lu_row_ = in_shape.at(kDim0);
      lu_col_ = lu_row_;
    } else if (in_shape.size() == kLuNormalShape) {
      lu_row_ = in_shape.at(kDim0);
      lu_col_ = in_shape.at(kDim1);
    } else {
      MS_LOG(ERROR) << "Input Only support Rank 1 OR 2";
      return false;
    }
    if (lu_row_ != lu_col_) {
      MS_LOG(ERROR) << "Cholesky need square matrix as input.";
      return false;
    }
    // set matrix row or col to be lead dimension
    m_ = SizeToInt(lu_row_);
    lda_ = m_;
    ldb_ = m_;
    InitSizeLists();
    return true;
  }

  void InitSizeLists() override {
    size_t input_size = lda_ * m_ * unit_size_;
    input_size_list_.push_back(input_size);

    size_t output_size = lda_ * m_ * unit_size_;
    size_t output_piv_size = 0;
    size_t output_info_size = sizeof(int);
    if (pivot_on_) {
      output_piv_size = m_ * sizeof(int);
    }
    output_size_list_.resize(kDim3);
    output_size_list_[kDim0] = output_size;
    output_size_list_[kDim1] = output_piv_size;
    output_size_list_[kDim2] = output_info_size;
  }

  size_t unit_size_{sizeof(T)};
  size_t lu_row_{0};
  size_t lu_col_{0};
  size_t m_{0};
  size_t lda_{0};
  size_t ldb_{0};
  int lwork_{0};
  bool pivot_on_{true};
  T *d_work_{nullptr};
  cusolverDnHandle_t handle_{nullptr};
  std::vector<size_t> input_size_list_{};
  std::vector<size_t> output_size_list_{};
  std::vector<size_t> workspace_size_list_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
