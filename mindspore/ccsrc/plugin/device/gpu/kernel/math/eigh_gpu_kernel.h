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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <map>
#include <string>
#include <type_traits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/triangle_matrix_copy_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "mindspore/core/ops/eigh.h"

namespace mindspore {
namespace kernel {
template <typename T>
class EighGpuKernelMod : public NativeGpuKernelMod {
 public:
  EighGpuKernelMod() : is_null_input_(false) {}
  ~EighGpuKernelMod() = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->GetPrim()->name();
    dtype_ = inputs[0]->GetDtype();

    auto kernel_ptr = std::dynamic_pointer_cast<ops::Eigh>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', kernel_ptr is null.";
      return false;
    }
    compute_eigen_vectors_ = kernel_ptr->get_compute_eigen_vectors();
    lower_ = kernel_ptr->get_lower();
    if (compute_eigen_vectors_) {
      jobz_ = CUSOLVER_EIG_MODE_VECTOR;
    } else {
      jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
    }
    cusolver_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    auto ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    auto A_shape = inputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(A_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_RESIZE_FAILED;
    }

    m_ = LongToSizeClipNeg(A_shape[0]);
    InitSizeLists();
    return KRET_OK;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(cusolver_handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "CusolverDnSetStream failed");
    // Matrix A, input or output(eigenvector)
    auto inout_A_addr = GetDeviceAddress<T>(inputs, kDim0);
    // Notice :this is important
    // A col or row major is different to cpu, so a lower triangle is a upper triangle, a upper is a lower in gpu mem
    // So the upper is positive to it from var, but for real scalar matrix, upper eq lower, it's different from complex
    if (lower_) {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    }
    auto output_addr = GetDeviceAddress<T>(outputs, kDim0);  // output eigenvalues
    // Output eigenvector if need
    T *output_v_addr = nullptr;
    if (compute_eigen_vectors_) {
      output_v_addr = GetDeviceAddress<T>(outputs, kDim1);  // output eigenvalues
    } else {
      output_v_addr = GetDeviceAddress<T>(workspace, kDim4);  // not output eigenvalues, use workspace
    }
    int *devInfo = GetDeviceAddress<int>(workspace, kDim0);
    auto w_v_addr = GetDeviceAddress<T>(workspace, kDim1);  // temp eigenvector before transpose
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(w_v_addr, inout_A_addr, m_ * m_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy to input matrix failed");
    size_t lda_ = m_;
    int lwork = 0;
    if constexpr (std::is_same_v<T, float>) {
      cusolverDnSsyevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_, inout_A_addr, lda_, output_addr, &lwork);
    } else if constexpr (std::is_same_v<T, double>) {
      cusolverDnDsyevd_bufferSize(cusolver_handle_, jobz_, uplo_, m_, inout_A_addr, lda_, output_addr, &lwork);
    }

    void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
    if constexpr (std::is_same_v<T, float>) {
      cusolverDnSsyevd(cusolver_handle_, jobz_, uplo_, m_, w_v_addr, lda_, output_addr, reinterpret_cast<T *>(d_work),
                       lwork, devInfo);
    } else if constexpr (std::is_same_v<T, double>) {
      cusolverDnDsyevd(cusolver_handle_, jobz_, uplo_, m_, w_v_addr, lda_, output_addr, reinterpret_cast<T *>(d_work),
                       lwork, devInfo);
    }
    size_t input_shape[kShape2dDims] = {m_, m_};
    size_t input_axis[kShape2dDims] = {1, 0};
    size_t *dev_input_shape = GetDeviceAddress<size_t>(workspace, kDim2);
    size_t *dev_input_axis = GetDeviceAddress<size_t>(workspace, kDim3);
    cudaMemcpyAsync(dev_input_shape, input_shape, kShape2dDims * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    cudaMemcpyAsync(dev_input_axis, input_axis, kShape2dDims * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    if (compute_eigen_vectors_) {
      CalTranspose(m_ * m_, w_v_addr, dev_input_shape, dev_input_axis, kShape2dDims, output_v_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
    int info_gpu = 0;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "Copy to device result failed");
    if (info_gpu != 0) {
      MS_LOG_EXCEPTION << kernel_name_ << " launch gpu kernel fail for dtype:" << dtype_;
    }
    return true;
  }

 protected:
  void InitSizeLists() {
    // Eigenvector if need
    workspace_size_list_.push_back(sizeof(int));
    workspace_size_list_.push_back(m_ * m_ * sizeof(T));
    // Transpose scalar workspace
    workspace_size_list_.push_back(kShape2dDims * sizeof(size_t));
    workspace_size_list_.push_back(kShape2dDims * sizeof(size_t));
    // A temp space for input/eigenvectors if eigenvector not need to output
    if (!compute_eigen_vectors_) {
      workspace_size_list_.push_back(m_ * m_ * sizeof(T));
    }
  }

  size_t m_{1};
  TypeId dtype_{kNumberTypeFloat32};
  cusolverDnHandle_t cusolver_handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  cusolverEigMode_t jobz_ = CUSOLVER_EIG_MODE_NOVECTOR;
  bool compute_eigen_vectors_{false};
  bool lower_{true};
  bool is_null_input_;
  std::vector<T *> h_array_{};
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EIGH_GPU_KERNEL_H
