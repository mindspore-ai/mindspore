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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
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
#include "utils/complex.h"
#include "backend/kernel_compiler/gpu/cuda_impl/real_to_complex_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr char C_EIEH_VECTOR[] = "compute_eigenvectors";
constexpr char LOWER[] = "lower";
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
struct Complex_traits {};

template <typename T>
struct Complex_traits<Complex<T>> {
  using value_type = T;
};

template <typename T>
class EighcGpuKernel : public GpuKernel {
 public:
  EighcGpuKernel() = default;
  ~EighcGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Init(const CNodePtr &kernel_node) override {
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    compute_eigen_vectors_ = static_cast<bool>(GetAttr<bool>(kernel_node, C_EIEH_VECTOR));
    lower_ = static_cast<bool>(GetAttr<bool>(kernel_node, LOWER));
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
    if (A_shape.size() != kShape2dDims || A_shape[1] != A_shape[1]) {
      MS_LOG(EXCEPTION) << "wrong array shape, A should be a square matrix, but got [" << A_shape[0] << " X "
                        << A_shape[1] << "]";
    }
    m_ = A_shape[0];
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(cusolver_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "cusolverDnSetStream failed");
    // matrix A, input or output(eigenvector)
    auto inout_A_addr = GetDeviceAddress<T>(inputs, kDim0);
    if (lower_) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }
    size_t lda_ = m_;
    auto output_w_addr = GetDeviceAddress<T>(outputs, kDim0);
    // output eigenvector
    auto output_v_addr = GetDeviceAddress<T>(outputs, kDim1);
    int *devInfo = GetDeviceAddress<int>(workspace, kDim0);
    // temp output eigenvalues real scalar
    auto w_w_addr = GetDeviceAddress<D>(workspace, kDim1);
    auto w_w_c_addr = GetDeviceAddress<T>(workspace, kDim2);
    // temp eigenvector before transpose
    auto w_v_addr = GetDeviceAddress<T>(workspace, kDim3);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_v_addr, inout_A_addr, m_ * m_ * sizeof(T),
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy input matrix failed");
    size_t input_shape[kShape2dDims] = {m_, m_};
    size_t input_axis[kShape2dDims] = {1, 0};
    size_t *dev_input_shape = GetDeviceAddress<size_t>(workspace, kDim4);
    size_t *dev_input_axis = GetDeviceAddress<size_t>(workspace, kDim5);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_input_shape, input_shape, kShape2dDims * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "malloc input shape workspace failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_input_axis, input_axis, kShape2dDims * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "malloc input shape workspace failed");
    CalTranspose(m_ * m_, output_v_addr, dev_input_shape, dev_input_axis, kShape2dDims, w_v_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    int lwork = 0;
    void *d_work = nullptr;
    if constexpr (std::is_same_v<T, Complex<float>>) {
      cusolverDnCheevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_, reinterpret_cast<cuComplex *>(output_v_addr),
                                  lda_, w_w_addr, &lwork);
    } else {
      cusolverDnZheevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_,
                                  reinterpret_cast<cuDoubleComplex *>(output_v_addr), lda_, w_w_addr, &lwork);
    }
    d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
    if (!d_work) {
      MS_LOG(EXCEPTION) << "GPU memory alloca failed.";
    }
    if constexpr (std::is_same_v<T, Complex<float>>) {
      cusolverDnCheevd(cusolver_handle_, jobz_, uplo_, m_, reinterpret_cast<cuComplex *>(w_v_addr), lda_, w_w_addr,
                       reinterpret_cast<cuComplex *>(d_work), lwork, devInfo);
    } else {
      cusolverDnZheevd(cusolver_handle_, jobz_, uplo_, m_, reinterpret_cast<cuDoubleComplex *>(w_v_addr), lda_,
                       w_w_addr, reinterpret_cast<cuDoubleComplex *>(d_work), lwork, devInfo);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(w_w_c_addr, w_w_addr, m_ * sizeof(D), cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy eigenvalue from workspace to host failed");
    // convert real scalar to complex
    RealToComplex(m_, reinterpret_cast<D *>(w_w_c_addr), reinterpret_cast<D *>(output_w_addr),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    CalTranspose(m_ * m_, w_v_addr, dev_input_shape, dev_input_axis, kShape2dDims, output_v_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
    int info_gpu = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "copy eigenvalues to outpu failed");
    if (info_gpu != 0) {
      MS_LOG_EXCEPTION << kernel_name_ << " launch gpu kernel fail for dtype:" << dtype_;
    }
    return true;
  }

 protected:
  void InitSizeLists() override {
    // in/out matrix, eigenvector
    input_size_list_.push_back(m_ * m_ * sizeof(T));
    // eigenvalues, cuda output original real scalar, should covert to complex<ft32/64>
    output_size_list_.push_back(m_ * sizeof(T));
    output_size_list_.push_back(m_ * m_ * sizeof(T));
    // result
    workspace_size_list_.push_back(sizeof(int));
    // for temp original eigenvalue real scalar
    workspace_size_list_.push_back(m_ * sizeof(D));
    // for temp pre-transpose complex mitrx
    workspace_size_list_.push_back(m_ * sizeof(T));
    workspace_size_list_.push_back(m_ * m_ * sizeof(T));
    // transpose scalar workspace
    workspace_size_list_.push_back(kShape2dDims * sizeof(size_t));
    workspace_size_list_.push_back(kShape2dDims * sizeof(size_t));
  }

  size_t m_{1};
  TypeId dtype_{kNumberTypeFloat32};
  cublasHandle_t blas_handle_{nullptr};
  cusolverDnHandle_t cusolver_handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  cusolverEigMode_t jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
  bool compute_eigen_vectors_{false};
  bool lower_{true};
  std::vector<T *> h_array_{};
  std::vector<size_t> input_size_list_{};
  std::vector<size_t> output_size_list_{};
  std::vector<size_t> workspace_size_list_{};
  using D = typename Complex_traits<T>::value_type;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
