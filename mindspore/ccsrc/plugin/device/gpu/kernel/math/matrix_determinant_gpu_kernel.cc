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
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/determinant_by_lu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
bool MatrixDeterminantGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  // A Code Block For getting launch_kernel function.
  {
    kernel_name_ = base_operator->name();
    if (kernel_name_ == prim::kPrimMatrixDeterminant->name()) {
      is_sign_log_determinant_ = false;
    } else if (kernel_name_ == prim::kPrimLogMatrixDeterminant->name()) {
      is_sign_log_determinant_ = true;
    } else {
      MS_LOG(ERROR) << "For 'MatrixDeterminant' or 'LogMatrixDeterminant' does not support, but got kernel name: "
                    << kernel_name_;
      return false;
    }
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  }

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  cublas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  return true;
}

bool MatrixDeterminantGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  DestroyResource();
  ResetResource();
  // For input shape and output shape's rationality have been checked in core/ops/matrix_determinant or
  // core/ops/log_matrix_determinant , we ignore shape's checking.
  input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                     inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  is_null_input_ = (input_elements_ == 0);
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  // For log_matrix_determinant, there are two outputs, but shapes are equal.
  output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_elements_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  constexpr size_t last_two_dims = 2;
  // Ignore last two dims <--> Inner [M, M]
  for (size_t i = 0; i < (input_shape_.size() - last_two_dims); ++i) {
    batch_size_ *= input_shape_.at(i);
  }
  m_ = input_shape_.back();
  InitSizeLists();
  outputs_ = outputs;
  return true;
}

template <typename T>
bool MatrixDeterminantGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  // For Lu factorization will inplace input data to be output.
  auto middle_lu_output = GetDeviceAddress<T>(workspace, kIndex0);
  auto batch_lu_device_address = GetDeviceAddress<T *>(workspace, kIndex1);
  auto pivot = GetDeviceAddress<int>(workspace, kIndex2);
  auto info = GetDeviceAddress<int>(workspace, kIndex3);
  auto device_input_shape = GetDeviceAddress<size_t>(workspace, kIndex4);
  auto device_input_axis = GetDeviceAddress<size_t>(workspace, kIndex5);

  std::vector<T *> batch_lu_address_data;
  for (size_t i = 0; i < batch_size_; i++) {
    batch_lu_address_data.emplace_back(middle_lu_output + i * m_ * m_);
  }

  // Transpose input data from rowMajor to colMajor.
  constexpr size_t input_shape_length = 3;
  std::vector<size_t> host_input_shape = {batch_size_, m_, m_};
  // From (0, 1, 2) --> (0, 2, 1)
  std::vector<size_t> host_input_axis = {0, 2, 1};
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(device_input_shape, host_input_shape.data(), sizeof(size_t) * input_shape_length,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(device_input_axis, host_input_axis.data(), sizeof(size_t) * input_shape_length,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  CalTranspose(input_elements_, input, device_input_shape, device_input_axis, input_shape_length, middle_lu_output,
               reinterpret_cast<cudaStream_t>(cuda_stream_));

  // Compute the partial pivoted lu factorization.
  // If m_ / batch_size_ <= 128 :
  //  We use batched cublas api is faster by empiricism, for small matrices or large batch.
  // Otherwise:
  // We use no-batched cusolver api is faster by empiricism, For small batch sizes. But ms cuda10 do not support
  // api cusolverDnXgetrf, just skip it.
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(batch_lu_device_address, batch_lu_address_data.data(), sizeof(T *) * batch_size_,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(cublas_handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For MatrixDeterminantGpuKernelMod cublasSetStream Fail");
  if (std::is_same<T, float>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasSgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<float **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "For MatrixDeterminantGpuKernelMod cublasSgetrfBatched Fail");
  } else if (std::is_same<T, double>::value) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasDgetrfBatched(cublas_handle_, SizeToInt(m_), reinterpret_cast<double **>(batch_lu_device_address),
                          SizeToInt(m_), pivot, info, SizeToInt(batch_size_)),
      "For MatrixDeterminantGpuKernelMod cublasDgetrfBatched Fail");
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input data type must be float or double.";
  }
  int host_info;
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost,
                                                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                    "For MatrixDeterminantGpuKernelMod cudaMemcpyAsync Fail");
  if (host_info > 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' " << host_info
                      << "-th parameter is wrong, please check your input data info.";
  }
  // Compute the determinant (-1)^s * prod(diag(U)), s is the order of the permutation in pivots and U is the result of
  // LU factorization.
  auto sign_output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  if (is_sign_log_determinant_) {
    // For LogMatrixDeterminant, two output -->(sign determinant, log_abs_determinant)
    auto log_determinant_output = reinterpret_cast<T *>(outputs.at(kIndex1)->addr);
    // For LogMatrixDeterminant, only one output -->(determinant)
    CalculateDeterminantByLu(middle_lu_output, pivot, SizeToInt(m_), SizeToInt(batch_size_), is_sign_log_determinant_,
                             log_determinant_output, sign_output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    // For MatrixDeterminant, only one output -->(determinant)
    auto determinant_output = sign_output;
    sign_output = nullptr;
    CalculateDeterminantByLu(middle_lu_output, pivot, SizeToInt(m_), SizeToInt(batch_size_), is_sign_log_determinant_,
                             determinant_output, sign_output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDeterminantGpuKernelMod::MatrixDeterminantFunc>>
  MatrixDeterminantGpuKernelMod::func_list_ = {
    // MatrixDeterminant's launch kernel
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<double>},
    // LogMatrixDeterminant's launch kernel
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> MatrixDeterminantGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDeterminantFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixDeterminant, MatrixDeterminantGpuKernelMod);
// Whether to computes the sign and the log of the absolute value of the determinant.
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogMatrixDeterminant, MatrixDeterminantGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
