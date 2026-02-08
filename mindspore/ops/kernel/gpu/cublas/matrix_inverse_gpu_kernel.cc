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

#include "kernel/gpu/cublas/matrix_inverse_gpu_kernel.h"
#include <map>
#include <utility>
#include <algorithm>
#include "mindspore/ops/infer/matrix_inverse.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool MatrixInverseGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;

  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  handle_cus = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  adjoint_ = GetValue<bool>(primitive_->GetAttr("adjoint"));
  if (adjoint_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' adjoint=True is not supported on GPU.";
  }
  return true;
}

int MatrixInverseGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  size_t kMinDim = 2;
  if (input_shape.size() < kMinDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
                      << input_shape.size();
  }
  size_t last_index = input_shape.size() - 1;
  if (input_shape[last_index] != input_shape[last_index - 1]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the last two dimensions of the input matrix should be equal, "
                      << "but got one: " << input_shape[last_index] << ", another: " << input_shape[last_index - 1];
  }
  size_ = input_shape[last_index];
  batch_size_ = 1;
  for (size_t i = 0; i < last_index - 1; i++) {
    batch_size_ *= input_shape[i];
  }
  auto dtype = inputs[kIndex0]->dtype_id();
  int typesize = GetTypeByte(TypeIdToType(dtype));
  dtype_size_ = sizeof(TypeIdToType(dtype));
  input_size_ = typesize;
  for (auto dim : input_shape) {
    input_size_ *= dim;
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
void MatrixInverseGpuKernelMod::LaunchKernel_Cublas(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &workspace,
                                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  auto compute_input_addr = GetDeviceAddress<T>(workspace, 0);
  auto lu_batch_addr = GetDeviceAddress<T *>(workspace, 1);
  auto inv_batch_addr = GetDeviceAddress<T *>(workspace, 2);
  auto pivo_addr = GetDeviceAddress<int>(workspace, 3);
  auto info_addr = GetDeviceAddress<int>(workspace, 4);
  int len = SizeToInt(size_);
  int batchsize = SizeToInt(batch_size_);
  std::vector<T *> lu_addr(batch_size_);
  std::vector<T *> inv_addr(batch_size_);
  for (size_t i = 0; i < batch_size_; i++) {
    lu_addr[i] = compute_input_addr + i * len * len;
    inv_addr[i] = output_addr + i * len * len;
  }
  CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cublasSetStream failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(compute_input_addr, input_addr, input_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cuda memcopy Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(lu_batch_addr, lu_addr.data(), sizeof(T *) * batch_size_,
                                                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                    "cuda memcopy Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(inv_batch_addr, inv_addr.data(), sizeof(T *) * batch_size_,
                                                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                    "cuda memcopy Fail");
  if (std::is_same<T, float>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSgetrfBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr),
                                                             len, pivo_addr, info_addr, batchsize),
                                         "cublas trsm batched Fail");
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasSgetriBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr), len, pivo_addr,
                          reinterpret_cast<float **>(inv_batch_addr), len, info_addr, batchsize),
      "cublas trsm batched Fail");
  } else if (std::is_same<T, double>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasDgetrfBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr),
                                                             len, pivo_addr, info_addr, batchsize),
                                         "cublas trsm batched Fail");
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasDgetriBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr), len, pivo_addr,
                          reinterpret_cast<double **>(inv_batch_addr), len, info_addr, batchsize),
      "cublas trsm batched Fail");
  } else if (std::is_same<T, cuComplex>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasCgetrfBatched(handle_, len, reinterpret_cast<cuComplex **>(lu_batch_addr), len, pivo_addr, info_addr,
                          batchsize),
      "cublas trsm batched Fail");
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasCgetriBatched(handle_, len, reinterpret_cast<cuComplex **>(lu_batch_addr), len, pivo_addr,
                          reinterpret_cast<cuComplex **>(inv_batch_addr), len, info_addr, batchsize),
      "cublas trsm batched Fail");
  } else if (std::is_same<T, cuDoubleComplex>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasZgetrfBatched(handle_, len, reinterpret_cast<cuDoubleComplex **>(lu_batch_addr), len, pivo_addr, info_addr,
                          batchsize),
      "cublas trsm batched Fail");
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasZgetriBatched(handle_, len, reinterpret_cast<cuDoubleComplex **>(lu_batch_addr), len, pivo_addr,
                          reinterpret_cast<cuDoubleComplex **>(inv_batch_addr), len, info_addr, batchsize),
      "cublas trsm batched Fail");
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type entered must be float or double or complex.";
  }
}

template <typename T>
void MatrixInverseGpuKernelMod::LaunchKernel_CuSolve(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &workspace,
                                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  auto compute_input_addr = GetDeviceAddress<T>(workspace, 0);
  int *info_output_addr = GetDeviceAddress<int>(workspace, 5);
  std::vector<T> di_addr_(input_size_);
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_cus, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cusolverDnSetStream failed");
  int len = SizeToInt(size_);
  if constexpr (std::is_same_v<T, cuComplex>) {
    for (int i = 0; i < len; ++i) {
      di_addr_[i * len + i] = make_cuComplex(1.0, 0.0);
    }
  }
  if constexpr (std::is_same_v<T, cuDoubleComplex>) {
    for (int i = 0; i < len; ++i) {
      di_addr_[i * len + i] = make_cuDoubleComplex(1.0, 0.0);
    }
  }
  if constexpr ((std::is_same_v<T, float>) || (std::is_same_v<T, double>)) {
    for (int i = 0; i < len; ++i) {
      di_addr_[i * len + i] = 1.0;
    }
  }
  int lwork = 0;
  T *d_work = nullptr;
  cudaMemcpyAsync(compute_input_addr, di_addr_.data(), input_size_, cudaMemcpyHostToDevice,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
  cudaMemcpyAsync(output_addr, compute_input_addr, input_size_, cudaMemcpyDeviceToDevice,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgetrf_bufferSize(handle_cus, len, len, input_addr, len, &lwork),
                                           "cusolverDnSgetrf_bufferSize failed");
    d_work = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork));
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSgetrf(handle_cus, len, len, input_addr, len, d_work, NULL, info_output_addr),
      "cusolverDnSgetrf failed");
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSgetrs(handle_cus, CUBLAS_OP_N, len, len, input_addr, len, NULL, output_addr, len, info_output_addr),
      "cusolverDnSgetrs failed");
  }
  if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgetrf_bufferSize(handle_cus, len, len, input_addr, len, &lwork),
                                           "cusolverDnSgetrf_bufferSize failed");
    d_work = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork));
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDgetrf(handle_cus, len, len, input_addr, len, d_work, NULL, info_output_addr),
      "cusolverDnDgetrf failed");
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDgetrs(handle_cus, CUBLAS_OP_N, len, len, input_addr, len, NULL, output_addr, len, info_output_addr),
      "cusolverDnDgetrs failed");
  }
  if constexpr (std::is_same_v<T, cuComplex>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnCgetrf_bufferSize(handle_cus, len, len, input_addr, len, &lwork),
                                           "cusolverDnCgetrf_bufferSize failed");
    d_work = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork));
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnCgetrf(handle_cus, len, len, input_addr, len, d_work, NULL, info_output_addr),
      "cusolverDnCgetrf failed");
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnCgetrs(handle_cus, CUBLAS_OP_N, len, len, input_addr, len, NULL, output_addr, len, info_output_addr),
      "cusolverDnCgetrs failed");
  }
  if constexpr (std::is_same_v<T, cuDoubleComplex>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnZgetrf_bufferSize(handle_cus, len, len, input_addr, len, &lwork),
                                           "cusolverDnZgetrf_bufferSize failed");
    d_work = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork));
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnZgetrf(handle_cus, len, len, input_addr, len, d_work, NULL, info_output_addr),
      "cusolverDnZgetrf failed");
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnZgetrs(handle_cus, CUBLAS_OP_N, len, len, input_addr, len, NULL, output_addr, len, info_output_addr),
      "cusolverDnZgetrs failed");
  }
  if (d_work != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
  }
  return;
}

template <typename T>
bool MatrixInverseGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  int len = SizeToInt(size_);
  int kNumber32 = 32;
  if (len < kNumber32 || batch_size_ > 1) {
    LaunchKernel_Cublas<T>(inputs, workspace, outputs, stream_ptr);
  } else {
    LaunchKernel_CuSolve<T>(inputs, workspace, outputs, stream_ptr);
  }
  return true;
}

void MatrixInverseGpuKernelMod::InitSizeLists() {
  workspace_size_list_.emplace_back(input_size_);
  size_t lu_size = batch_size_ * dtype_size_;
  workspace_size_list_.emplace_back(lu_size);
  size_t inv_size = batch_size_ * dtype_size_;
  workspace_size_list_.emplace_back(inv_size);
  size_t pivo_size = batch_size_ * size_ * sizeof(int);
  workspace_size_list_.emplace_back(pivo_size);
  size_t info_size = batch_size_ * sizeof(int);
  workspace_size_list_.emplace_back(info_size);
  workspace_size_list_.emplace_back(sizeof(int));
}

std::vector<std::pair<KernelAttr, MatrixInverseGpuKernelMod::MatrixInverseFunc>> MatrixInverseGpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &MatrixInverseGpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &MatrixInverseGpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
    &MatrixInverseGpuKernelMod::LaunchKernel<cuComplex>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    &MatrixInverseGpuKernelMod::LaunchKernel<cuDoubleComplex>}};

std::vector<KernelAttr> MatrixInverseGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixInverseFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixInverse, MatrixInverseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
