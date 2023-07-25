/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/matrix_determinant_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/ops/math_ops.h"
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/determinant_by_lu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"

namespace mindspore {
namespace kernel {
bool MatrixDeterminantGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (kernel_name_ == prim::kPrimMatrixDeterminant->name()) {
    is_sign_log_determinant_ = false;
  } else if (kernel_name_ == prim::kPrimLogMatrixDeterminant->name()) {
    is_sign_log_determinant_ = true;
  } else {
    MS_LOG(ERROR) << "For 'MatrixDeterminant' or 'LogMatrixDeterminant', it does not support, but got kernel name: "
                  << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  cublas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  return true;
}

int MatrixDeterminantGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  // For input shape and output shape's rationality have been checked in core/ops/matrix_determinant or
  // core/ops/log_matrix_determinant , we ignore shape's checking.
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  input_shape_.clear();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input shape");
  if (is_null_input_) {
    return KRET_OK;
  }
  constexpr size_t last_two_dims = 2;
  // Ignore last two dims <--> Inner [M, M]
  batch_size_ = 1;
  for (size_t i = 0; i < (input_shape_.size() - last_two_dims); ++i) {
    batch_size_ *= input_shape_.at(i);
  }
  m_ = input_shape_.back();
  InitWorkSpaceSizeList();
  return KRET_OK;
}

template <typename T>
bool MatrixDeterminantGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  // For Lu factorization will inplace input data to be output.
  auto middle_lu_output = GetDeviceAddress<T>(workspace, kIndex0);
  auto batch_lu_device_address = GetDeviceAddress<T *>(workspace, kIndex1);
  auto pivot = GetDeviceAddress<int>(workspace, kIndex2);
  auto info = GetDeviceAddress<int>(workspace, kIndex3);
  std::vector<T *> batch_lu_address_data;
  for (size_t i = 0; i < batch_size_; i++) {
    batch_lu_address_data.emplace_back(middle_lu_output + i * m_ * m_);
  }
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(batch_lu_device_address, batch_lu_address_data.data(), sizeof(T *) * batch_size_,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(cublas_handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For MatrixDeterminantGpuKernelMod cublasSetStream Fail");
  // Transpose input data from rowMajor to colMajor.
  cudaError_t status = cudaErrorNotReady;
  status = MatrixTranspose(input, SizeToInt(input_elements_), SizeToInt(m_), SizeToInt(m_), middle_lu_output,
                           device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  // Compute the partial pivoted lu factorization.
  // If m_ / batch_size_ <= 128 :
  //  We use batched cublas api is faster by empiricism, for small matrices or large batch.
  // Otherwise:
  // We use no-batched cusolver api is faster by empiricism, For small batch sizes. But ms cuda10 do not support
  // api cusolverDnXgetrf, just skip it.
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasSgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<float **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "MatrixDeterminantGpuKernelMod cublasSgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasDgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<double **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "MatrixDeterminantGpuKernelMod cublasDgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, utils::Complex<float>>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasCgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<cuComplex **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "MatrixDeterminantGpuKernelMod cublasCgetrfBatched Fail");
  } else if constexpr (std::is_same_v<T, utils::Complex<double>>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasZgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<cuDoubleComplex **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "MatrixDeterminantGpuKernelMod cublasZgetrfBatched Fail");
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', it's the input data type must be float32, float64, complex64 or complex128.";
    return false;
  }
  // Just checking first of lu factorization info is ok.
  int host_info;
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost,
                                                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                    "For MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  // Sync host info
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cudaStreamSynchronized failed");
  if (host_info > 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', it's " << host_info
                    << "-th parameter is wrong, please check your input data info.";
  }
  // Compute the determinant (-1)^s * prod(diag(U)), s is the order of the permutation in pivots and U is the result of
  // LU factorization.
  auto sign_output = GetDeviceAddress<T>(outputs, kIndex0);
  if (is_sign_log_determinant_) {
    // For LogMatrixDeterminant, two output -->(sign determinant, log_abs_determinant)
    auto log_determinant_output = GetDeviceAddress<T>(outputs, kIndex1);
    status = CalculateDeterminantByLu(middle_lu_output, pivot, SizeToInt(m_), SizeToInt(batch_size_),
                                      is_sign_log_determinant_, log_determinant_output, sign_output, device_id_,
                                      reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    // For MatrixDeterminant, only one output -->(determinant)
    auto determinant_output = sign_output;
    sign_output = nullptr;
    status = CalculateDeterminantByLu(middle_lu_output, pivot, SizeToInt(m_), SizeToInt(batch_size_),
                                      is_sign_log_determinant_, determinant_output, sign_output, device_id_,
                                      reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDeterminantGpuKernelMod::MatrixDeterminantFunc>>
  MatrixDeterminantGpuKernelMod::func_list_ = {
    // MatrixDeterminant's launch kernel
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<utils::Complex<double>>},
};

std::vector<std::pair<KernelAttr, LogMatrixDeterminantGpuKernelMod::LogMatrixDeterminantFunc>>
  LogMatrixDeterminantGpuKernelMod::func_list_ = {
    // LogMatrixDeterminant's launch kernel
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddAllSameAttr(true),
     &LogMatrixDeterminantGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddAllSameAttr(true),
     &LogMatrixDeterminantGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64).AddAllSameAttr(true),
     &LogMatrixDeterminantGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128).AddAllSameAttr(true),
     &LogMatrixDeterminantGpuKernelMod::LaunchKernel<utils::Complex<double>>},
};

void MatrixDeterminantGpuKernelMod::InitWorkSpaceSizeList() {
  // The workspace for device middle lu output size.
  const size_t middle_output_size = input_elements_ * unit_size_;
  // The workspace for device batch lu device address.
  const size_t batch_lu_address_size = batch_size_ * sizeof(void *);
  // The workspace for device lu pivot size.
  const size_t pivot_size = batch_size_ * m_ * sizeof(int);
  // The workspace for device return info.
  const size_t info_size = batch_size_ * sizeof(int);
  workspace_size_list_.clear();
  workspace_size_list_ = {middle_output_size, batch_lu_address_size, pivot_size, info_size};
}

std::vector<KernelAttr> MatrixDeterminantGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDeterminantFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<KernelAttr> LogMatrixDeterminantGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LogMatrixDeterminantFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixDeterminant, MatrixDeterminantGpuKernelMod);
// Whether to computes the sign and the log of the absolute value of the determinant.
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogMatrixDeterminant, LogMatrixDeterminantGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
