/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cublas/lu_gpu_kernel.h"
#include <iostream>
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"

namespace mindspore {
namespace kernel {
bool LuGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  cublas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  return true;
}

int LuGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->dtype_id());
  auto in_shape = inputs.at(kIndex0)->GetShapeVector();
  if (in_shape.size() <= 1) {
    MS_LOG(ERROR) << kernel_name_ << " input shape is " << in_shape.size() << " which is invalid.";
    return KRET_RESIZE_FAILED;
  }
  constexpr size_t lu_reverse_row_dim = 2;
  m_ = static_cast<size_t>(in_shape.at(in_shape.size() - lu_reverse_row_dim));
  n_ = static_cast<size_t>(in_shape.at(in_shape.size() - 1));
  k_ = std::min(m_, n_);
  input_elements_ = SizeOf(in_shape);
  batch_size_ = 1;
  for (int batch = 0; batch < static_cast<int>(in_shape.size() - lu_reverse_row_dim); ++batch) {
    batch_size_ *= static_cast<size_t>(in_shape.at(batch));
  }

  // a device addr to place lu factor return code
  workspace_size_list_.push_back(sizeof(int));

  // transpose workspace
  workspace_size_list_.push_back(batch_size_ * m_ * n_ * unit_size_);
  workspace_size_list_.push_back(batch_size_ * n_ * sizeof(int));

  // The workspace for device return info.
  workspace_size_list_.push_back(batch_size_ * sizeof(void *));
  workspace_size_list_.push_back(batch_size_ * sizeof(int));
  return KRET_OK;
}

void LuGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
void LuGpuKernelMod::BufferSize(T *batch_output_addr, int *lwork) {
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgetrf_bufferSize(handle_, m_, n_, batch_output_addr, m_, lwork),
                                           "cusolver query lu work size fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgetrf_bufferSize(handle_, m_, n_, batch_output_addr, m_, lwork),
                                           "cusolver query lu work size fail");
  } else if constexpr (std::is_same_v<T, utils::Complex<float>>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnCgetrf_bufferSize(handle_, m_, n_, reinterpret_cast<cuComplex *>(batch_output_addr), m_, lwork),
      "cusolver query lu work size fail");
  } else {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnZgetrf_bufferSize(handle_, m_, n_, reinterpret_cast<cuDoubleComplex *>(batch_output_addr), m_, lwork),
      "cusolver query lu work size fail");
  }
}

template <typename T, typename S>
void LuGpuKernelMod::LaunchKernel_CuSolve(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs) {
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "cusolverDnSetStream failed");
  T *batch_input_addr = GetDeviceAddress<T>(inputs, kDim0);
  T *batch_output_addr = GetDeviceAddress<T>(outputs, kDim0);
  MS_EXCEPTION_IF_NULL(batch_input_addr);
  MS_EXCEPTION_IF_NULL(batch_output_addr);
  T *d_work_ = nullptr;
  S *batch_piv_output_addr = nullptr;
  if (pivot_on_) {
    batch_piv_output_addr = GetDeviceAddress<S>(outputs, kDim1);
    MS_EXCEPTION_IF_NULL(batch_piv_output_addr);
  }
  int *info_output_addr = GetDeviceAddress<int>(workspace, kDim0);
  T *dev_work = GetDeviceAddress<T>(workspace, kDim1);
  int *dev_batch_piv = GetDeviceAddress<int>(workspace, kDim2);
  MS_EXCEPTION_IF_NULL(info_output_addr);
  MS_EXCEPTION_IF_NULL(dev_work);
  MS_EXCEPTION_IF_NULL(dev_batch_piv);
  // query working space of getrf
  BufferSize(batch_output_addr, &lwork_);
  // Transpose input data from rowMajor to colMajor.
  auto status = MatrixTranspose(batch_input_addr, SizeToInt(input_elements_), m_, m_, dev_work, device_id_,
                                reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  // malloc device working space of getrf
  d_work_ = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(unit_size_ * lwork_));
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    S *piv_output_addr = batch_piv_output_addr + batch * k_;
    int *dev_piv = dev_batch_piv + batch * k_;
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnSgetrf(handle_, m_, n_, dev_work + batch * m_ * n_, m_, d_work_, dev_piv, info_output_addr),
        "cusolver lu fail");
    } else if constexpr (std::is_same_v<T, double>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnDgetrf(handle_, m_, n_, dev_work + batch * m_ * n_, m_, d_work_, dev_piv, info_output_addr),
        "cusolver lu fail");
    } else if constexpr (std::is_same_v<T, utils::Complex<float>>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnCgetrf(handle_, m_, n_, reinterpret_cast<cuComplex *>(dev_work + batch * m_ * n_), m_,
                         reinterpret_cast<cuComplex *>(d_work_), dev_piv, info_output_addr),
        "cusolver lu fail");
    } else {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnZgetrf(handle_, m_, n_, reinterpret_cast<cuDoubleComplex *>(dev_work + batch * m_ * n_), m_,
                         reinterpret_cast<cuDoubleComplex *>(d_work_), dev_piv, info_output_addr),
        "cusolver lu fail");
    }

    std::vector<int> host_permuted(k_, 0);
    std::vector<int> host_pivots(k_, 0);
    std::vector<S> host_p(k_, 0);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(host_pivots.data(), dev_piv, sizeof(int) * k_, cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For 'Lu', cudaMemcpyAsync failed in LuGpuKernelMod::Launch copy pivots to host.");
    if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                         "cuda Stream Sync Failed.");
    }
    // cal pivots && permutation major by row.
    for (size_t i = 0; i < k_; ++i) {
      host_pivots[i] -= 1;
      host_permuted[i] = i;
    }
    for (size_t i = 0; i < k_; ++i) {
      int tmp_value = host_permuted[i];
      host_permuted[i] = host_permuted[host_pivots[i]];
      host_permuted[host_pivots[i]] = tmp_value;
    }
    for (size_t i = 0; i < k_; ++i) {
      host_p[i] = host_permuted[i];
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(piv_output_addr, host_p.data(), sizeof(S) * k_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cudaMemcpyAsync failed in LuGpuKernelMod::Launch copy pivots array.");
  }
  status = MatrixTranspose(dev_work, SizeToInt(input_elements_), m_, m_, batch_output_addr, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, "MatrixTranspose called by " + kernel_name_);
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work_);
}

template <typename T, typename S>
void LuGpuKernelMod::LaunchKernel_Cublas(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *batch_input_addr = GetDeviceAddress<T>(inputs, kDim0);
  T *batch_output_addr = GetDeviceAddress<T>(outputs, kDim0);
  S *batch_piv_output_addr = nullptr;
  if (pivot_on_) {
    batch_piv_output_addr = GetDeviceAddress<S>(outputs, kDim1);
  }
  T *dev_transpose_work = GetDeviceAddress<T>(workspace, kDim1);
  auto dev_batch_piv = GetDeviceAddress<int>(workspace, kDim2);
  auto batch_lu_device_address = GetDeviceAddress<T *>(workspace, kDim3);
  auto info = GetDeviceAddress<int>(workspace, kDim4);
  std::vector<T *> batch_lu_address_data;
  for (size_t i = 0; i < batch_size_; i++) {
    batch_lu_address_data.emplace_back(dev_transpose_work + i * m_ * m_);
  }
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(batch_lu_device_address, batch_lu_address_data.data(), sizeof(T *) * batch_size_,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "LuGpuKernelMod cudaMemcpyAsync Fail");
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(cublas_handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For LuGpuKernelMod cublasSetStream Fail");
  // Transpose input data from rowMajor to colMajor.
  auto status = MatrixTranspose(batch_input_addr, SizeToInt(input_elements_), m_, m_, dev_transpose_work, device_id_,
                                reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, "MatrixTranspose called by " + kernel_name_);
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasSgetrfBatched(cublas_handle_, m_, reinterpret_cast<float **>(batch_lu_device_address), m_, dev_batch_piv,
                          info, SizeToInt(batch_size_)),
      "LuGpuKernelMod cublasSgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasDgetrfBatched(cublas_handle_, m_, reinterpret_cast<double **>(batch_lu_device_address), m_, dev_batch_piv,
                          info, SizeToInt(batch_size_)),
      "LuGpuKernelMod cublasDgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, utils::Complex<float>>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasCgetrfBatched(cublas_handle_, m_, reinterpret_cast<cuComplex **>(batch_lu_device_address), m_,
                          dev_batch_piv, info, SizeToInt(batch_size_)),
      "LuGpuKernelMod cublasCgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, utils::Complex<double>>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasZgetrfBatched(cublas_handle_, m_, reinterpret_cast<cuDoubleComplex **>(batch_lu_device_address), m_,
                          dev_batch_piv, info, SizeToInt(batch_size_)),
      "LuGpuKernelMod cublasZgetrfBatched Fail");
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', it's the input data type must be float32, float64, complex64 or complex128.";
  }
  status = MatrixTranspose(dev_transpose_work, SizeToInt(input_elements_), m_, m_, batch_output_addr, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, "MatrixTranspose called by " + kernel_name_);
  std::vector<int> host_permuted(batch_size_ * k_, 0);
  std::vector<int> host_pivots(batch_size_ * k_, 0);
  std::vector<S> host_p(batch_size_ * k_, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(host_pivots.data(), dev_batch_piv, sizeof(int) * batch_size_ * k_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'Lu', cudaMemcpyAsync failed in LuGpuKernelMod::Launch copy pivots to host.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "cuda Stream Sync Failed.");
  }
  for (size_t i = 0; i < batch_size_; ++i) {
    for (size_t j = 0; j < k_; ++j) {
      host_permuted[i * k_ + j] = j;
      host_pivots[i * k_ + j] -= 1;
    }
    for (size_t j = 0; j < k_; ++j) {
      int tmp_value = host_permuted[i * k_ + j];
      host_permuted[i * k_ + j] = host_permuted[i * k_ + host_pivots[i * k_ + j]];
      host_permuted[i * k_ + host_pivots[i * k_ + j]] = tmp_value;
    }
  }
  for (size_t i = 0; i < batch_size_; ++i) {
    for (size_t j = 0; j < k_; ++j) {
      host_p[i * k_ + j] = host_permuted[i * k_ + j];
    }
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(batch_piv_output_addr, host_p.data(), sizeof(S) * batch_size_ * k_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync failed in LuGpuKernelMod::Launch copy pivots array.");
}

template <typename T, typename S>
bool LuGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs) {
  // If m_ / batch_size_ <= 128 :
  // We use batched cublas api is faster by empiricism, for small matrices or large batch.
  // Otherwise:
  // We use no-batched cusolver api is faster by empiricism, For small batch sizes.
  const size_t kNumber128 = 128;
  if (m_ / batch_size_ <= kNumber128) {
    LaunchKernel_Cublas<T, S>(inputs, workspace, outputs);
  } else {
    LaunchKernel_CuSolve<T, S>(inputs, workspace, outputs);
  }
  return true;
}
const std::vector<std::pair<KernelAttr, LuGpuKernelMod::KernelRunFunc>> &LuGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, LuGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &LuGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
     &LuGpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32),
     &LuGpuKernelMod::LaunchKernel<utils::Complex<float>, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32),
     &LuGpuKernelMod::LaunchKernel<utils::Complex<double>, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &LuGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &LuGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt64),
     &LuGpuKernelMod::LaunchKernel<utils::Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt64),
     &LuGpuKernelMod::LaunchKernel<utils::Complex<double>, int64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Lu, LuGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
