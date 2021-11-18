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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <type_traits>
#include "backend/kernel_compiler/gpu/cuda_impl/triangle_matrix_copy_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
constexpr char C_EIEH_VECTOR[] = "compute_eigenvectors";
template <typename T>
class EighGpuKernel : public GpuKernel {
 public:
  EighGpuKernel() = default;
  ~EighGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Init(const CNodePtr &kernel_node) override {
    dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    compute_eigen_vectors_ = static_cast<bool>(GetAttr<bool>(kernel_node, C_EIEH_VECTOR));
    if (compute_eigen_vectors_) {
      jobz_ = CUSOLVER_EIG_MODE_VECTOR;
    } else {
      jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
    }
    cusolver_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    bool is_null_input = CHECK_NULL_INPUT(A_shape);
    if (is_null_input) {
      MS_LOG(EXCEPTION) << "For 'EighValue GpuKernel', input is null";
    }
    if (A_shape.size() != kShape2dDims || A_shape[0] != A_shape[1]) {
      MS_LOG(EXCEPTION) << "wrong array shape, A should be a square matrix, but got [" << A_shape[0] << " X "
                        << A_shape[1] << "]";
    }
    m_ = A_shape[0];
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    // matrix A, input or output(eigenvector)
    auto inout_A_addr = GetDeviceAddress<T>(inputs, 0);
    auto lower = GetDeviceAddress<bool>(inputs, 1);
    bool h_lower{true};
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&h_lower, lower, sizeof(bool), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy to lower to device failed");
    if (h_lower) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }
    auto output_addr = GetDeviceAddress<T>(outputs, 0);    // output eigenvalues
    auto output_v_addr = GetDeviceAddress<T>(outputs, 1);  // output eigenvalues
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_v_addr, inout_A_addr, m_ * m_ * sizeof(T),
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy to input matrix failed");
    size_t lda_ = m_;

    int lwork = 0;
    if constexpr (std::is_same_v<T, float>) {
      cusolverDnSsyevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_, inout_A_addr, lda_, output_addr, &lwork);
    } else if constexpr (std::is_same_v<T, double>) {
      cusolverDnDsyevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_, inout_A_addr, lda_, output_addr, &lwork);
    }
    int *devInfo = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int));
    T *d_work = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * lwork);
    if constexpr (std::is_same_v<T, float>) {
      cusolverDnSsyevd(cusolver_handle_, jobz_, uplo_, m_, output_v_addr, lda_, output_addr, d_work, lwork, devInfo);
    } else if constexpr (std::is_same_v<T, double>) {
      cusolverDnDsyevd(cusolver_handle_, jobz_, uplo_, m_, output_v_addr, lda_, output_addr, d_work, lwork, devInfo);
    }
    if (d_work) {
      cudaFree(d_work);
    }
    int info_gpu = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy to device result failed");
    if (devInfo) {
      cudaFree(devInfo);
    }
    if (info_gpu != 0) {
      MS_LOG_EXCEPTION << kernel_name_ << " launch gpu kernel fail for dtype:" << dtype_;
    }
    return true;
  }

 protected:
  void InitSizeLists() override {
    // in/out matrix, eigenvector
    input_size_list_.push_back(m_ * m_ * sizeof(T));
    // uplo
    input_size_list_.push_back(sizeof(bool));
    // eigenvalues
    output_size_list_.push_back(m_ * sizeof(T));
    // eigenvector
    output_size_list_.push_back(m_ * m_ * sizeof(T));
  }

  size_t m_{1};
  TypeId dtype_{kNumberTypeFloat32};
  cusolverDnHandle_t cusolver_handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  cusolverEigMode_t jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
  bool compute_eigen_vectors_{false};
  std::vector<T *> h_array_{};
  std::vector<size_t> input_size_list_{};
  std::vector<size_t> output_size_list_{};
  std::vector<size_t> workspace_size_list_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
