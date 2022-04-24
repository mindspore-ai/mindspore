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
#include "mindspore/core/ops/matrix_determinant.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/determinant_by_lu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
bool MatrixDeterminantGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  // A Code Block For getting launch_kernel function.
  {
    kernel_ptr_ = std::make_shared<ops::MatrixDeterminant>(base_operator->GetPrim());
    kernel_name_ = kernel_ptr_->name();
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

  // A Code Block For setting input and output shape.
  {
    // For input shape and output shape's rationality have been checked in core/ops/matrix_determinant, we ignore
    // shape's checking.
    input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
    input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
    is_null_input_ = (input_elements_ == 0);
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    outputs_ = outputs;
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
  }

  // A Code Block For dealing with input_dynamic_shape.
  {
    if (!is_input_dynamic_shape_.has_value()) {
      bool is_input_dynamic_shape = false;
      for (const auto &input : inputs) {
        auto input_shape = input->GetShapeVector();
        if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim < 0; })) {
          is_input_dynamic_shape = true;
          break;
        }
      }
      is_input_dynamic_shape_ = is_input_dynamic_shape;
    }
  }

  cublas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  cusolver_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  return true;
}

bool MatrixDeterminantGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  if (is_input_dynamic_shape_.has_value() && is_input_dynamic_shape_.value()) {
    DestroyResource();
    ResetResource();
    return Init(base_operator, inputs, outputs);
  } else {
    kernel_ptr_ = base_operator;
    outputs_ = outputs;
    return true;
  }
}

template <typename T>
bool MatrixDeterminantGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);

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
  constexpr size_t divided_line_by_empiricism = 128;
  if (m_ / batch_size_ <= divided_line_by_empiricism) {
    // For small matrices or large batch, use batched cublas api is faster by empiricism.
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
  } else {
    // For small batch sizes, use no-batched cusolver api is faster by empiricism.
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSetStream(cusolver_handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For MatrixDeterminantGpuKernelMod cusolverDnSetStream Fail");
    cusolverDnParams_t cusolver_parameter;
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnCreateParams(&cusolver_parameter),
                                           "For MatrixDeterminantGpuKernelMod cusolverDnCreateParams Fail");
    // We default use new Algo api.
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSetAdvOptions(cusolver_parameter, CUSOLVERDN_GETRF, CUSOLVER_ALG_0),
      "For MatrixDeterminantGpuKernelMod cusolverDnSetAdvOptions Fail");
    T *device_buffer = nullptr;
    T *host_buffer = nullptr;
    size_t host_buffer_size = 0;
    size_t device_buffer_size = 0;
    cudaDataType cuda_data_type = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;
    // Query work buffer space.
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnXgetrf_bufferSize(cusolver_handle_, cusolver_parameter, m_, m_, cuda_data_type, middle_lu_output, m_,
                                  cuda_data_type, &device_buffer_size, &host_buffer_size),
      "For MatrixDeterminantGpuKernelMod cusolverDnSetAdvOptions Fail");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMalloc(&device_buffer, device_buffer_size * unit_size_),
                                      "For MatrixDeterminantGpuKernelMod cudaMalloc Fail");
    for (size_t i = 0; i < batch_size_; ++i) {
      // For pivot value may not exceed then range of int, so casting to int64 directly is ok.
      cusolverDnXgetrf(cusolver_handle_, cusolver_parameter, SizeToLong(m_), SizeToLong(m_), cuda_data_type,
                       batch_lu_address_data[i], SizeToLong(m_), reinterpret_cast<int64_t *>(pivot), cuda_data_type,
                       device_buffer, device_buffer_size, host_buffer, host_buffer_size, info);
    }
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
  CalculateDeterminantByLu(middle_lu_output, pivot, SizeToInt(m_), SizeToInt(batch_size_), output,
                           reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDeterminantGpuKernelMod::MatrixDeterminantFunc>>
  MatrixDeterminantGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixDeterminantGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> MatrixDeterminantGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDeterminantFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixDeterminant, MatrixDeterminantGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
