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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <algorithm>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/real_to_complex_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/triangle_matrix_copy_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

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
class EighcGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  EighcGpuKernelMod() : is_null_input_(false) {}
  ~EighcGpuKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    auto A_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    compute_eigen_vectors_ = static_cast<bool>(GetAttr<bool>(kernel_node, C_EIEH_VECTOR));
    lower_ = static_cast<bool>(GetAttr<bool>(kernel_node, LOWER));
    if (compute_eigen_vectors_) {
      jobz_ = CUSOLVER_EIG_MODE_VECTOR;
    } else {
      jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
    }
    cusolver_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    is_null_input_ = CHECK_SHAPE_NULL(A_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (A_shape.size() != kShape2dDims) {
      MS_LOG(EXCEPTION) << "Wrong array shape. For '" << kernel_name_ << "', a should be 2D, but got ["
                        << A_shape.size() << "] dimensions.";
    }
    if (A_shape[kDim0] != A_shape[kDim1]) {
      MS_LOG(EXCEPTION) << "Wrong array shape, For '" << kernel_name_
                        << "', a should be a squre matrix like [N X N], but got shape [" << A_shape[kDim0] << " X "
                        << A_shape[kDim1] << "].";
    }
    m_ = LongToSizeClipNeg(A_shape[0]);
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "CublasSetStream failed");
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(cusolver_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "CusolverDnSetStream failed");
    // Matrix A, input or output(eigenvector)
    auto inout_A_addr = GetDeviceAddress<T>(inputs, kDim0);
    if (lower_) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }
    size_t lda_ = m_;
    auto output_w_addr = GetDeviceAddress<T>(outputs, kDim0);
    // Output eigenvector if need
    T *output_v_addr = nullptr;
    if (compute_eigen_vectors_) {
      // output eigenvalues
      output_v_addr = GetDeviceAddress<T>(outputs, kDim1);
    } else {
      // not output eigenvalues, use workspace
      output_v_addr = GetDeviceAddress<T>(workspace, kDim4);
    }
    int *devInfo = GetDeviceAddress<int>(workspace, kDim0);
    // Temp output eigenvalues real scalar
    auto w_w_addr = GetDeviceAddress<D>(workspace, kDim1);
    auto w_w_c_addr = GetDeviceAddress<T>(workspace, kDim2);
    // Temp eigenvector before transpose
    auto w_v_addr = GetDeviceAddress<T>(workspace, kDim3);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_v_addr, inout_A_addr, m_ * m_ * sizeof(T),
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Copy input matrix failed");

    TransposeInfo info;
    info.input_shape = std::vector<int64_t>{m_, m_};
    info.perm = std::vector<int32_t>{1, 0};
    auto s1 =
      CalTranspose<T, false>(m_ * m_, output_v_addr, info, w_v_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(s1, "Transpose called by " + kernel_name_);

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
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', GPU memory alloca failed.";
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
                               "Copy eigenvalue from workspace to host failed");
    // Convert real scalar to complex
    RealToComplex(m_, reinterpret_cast<D *>(w_w_c_addr), reinterpret_cast<D *>(output_w_addr),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    if (compute_eigen_vectors_) {
      auto s2 =
        CalTranspose<T, false>(m_ * m_, w_v_addr, info, output_v_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(s2, "Transpose called by " + kernel_name_);
    }
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
    int info_gpu = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "For 'EignC', copy eigenvalues to output failed");
    if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                         "For 'EignC', cuda Stream Sync Failed.");
    }
    if (info_gpu != 0) {
      MS_LOG_EXCEPTION << kernel_name_ << " launch gpu kernel fail for dtype:" << dtype_;
    }
    return true;
  }

 protected:
  void InitSizeLists() override {
    // In/out matrix, eigenvector
    input_size_list_.push_back(m_ * m_ * sizeof(T));
    // Eigenvalues, cuda output original real scalar, should convert to complex<ft32/64>
    output_size_list_.push_back(m_ * sizeof(T));
    // Eigenvector if need
    if (compute_eigen_vectors_) {
      output_size_list_.push_back(m_ * m_ * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(int));
    // For temp original eigenvalue real scalar
    workspace_size_list_.push_back(m_ * sizeof(D));
    // For temp pre-transpose complex mitrx
    workspace_size_list_.push_back(m_ * sizeof(T));
    workspace_size_list_.push_back(m_ * m_ * sizeof(T));
    // A temp space for input/eigenvectors if eigenvector not need to output
    if (!compute_eigen_vectors_) {
      workspace_size_list_.push_back(m_ * m_ * sizeof(T));
    }
  }

  int64_t m_{1};
  TypeId dtype_{kNumberTypeFloat32};
  cublasHandle_t blas_handle_{nullptr};
  cusolverDnHandle_t cusolver_handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  cusolverEigMode_t jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
  bool compute_eigen_vectors_{false};
  bool lower_{true};
  std::vector<T *> h_array_{};
  using D = typename Complex_traits<T>::value_type;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_C_GPU_KERNEL_H
