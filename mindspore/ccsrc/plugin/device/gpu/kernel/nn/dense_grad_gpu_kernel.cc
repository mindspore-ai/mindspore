/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/dense_grad_gpu_kernel.h"
#include <map>
#include <algorithm>
#include <utility>
#include <memory>
#include "ops/nn_op_name.h"
#include "ops/grad/dense_grad.h"

namespace mindspore {
namespace kernel {
bool DenseGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_vec_[index].second;

  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  dtype_x_ = GetCudaDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
  dtype_w_ = GetCudaDataType(TypeIdLabel(inputs[kIndex1]->GetDtype()));
  dtype_dout_ = GetCudaDataType(TypeIdLabel(inputs[kIndex2]->GetDtype()));
  dtype_dx_ = GetCudaDataType(TypeIdLabel(outputs[kIndex0]->GetDtype()));
  dtype_dw_ = GetCudaDataType(TypeIdLabel(outputs[kIndex1]->GetDtype()));

  if (dtype_x_ != dtype_w_ || dtype_x_ != dtype_dout_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the types of inputs are not the same.";
  }
  if (dtype_x_ == CUDA_R_16F && dtype_w_ == CUDA_R_16F && dtype_dout_ == CUDA_R_16F && dtype_dx_ == CUDA_R_16F &&
      dtype_dw_ == CUDA_R_16F) {
    MS_LOG(INFO) << "input and output type is float16, allow to use Tensor Core operations if possible";
    algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  has_bias_ = GetValue<bool>(base_operator->GetAttr("has_bias"));
  if (has_bias_) {
    InitResource();
  }

  return true;
}

#if CUDA_VERSION >= 11000
cublasComputeType_t DenseGradGpuKernelMod::GetComputeType() {
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (dtype_x_ == CUDA_R_8I && dtype_dx_ == CUDA_R_32I) {
    compute_type = CUBLAS_COMPUTE_32I;
  } else if (dtype_x_ == CUDA_R_16F || dtype_x_ == CUDA_R_32F || dtype_dx_ == CUDA_R_32F) {
    compute_type = CUBLAS_COMPUTE_32F;
  } else if (dtype_x_ == CUDA_R_64F) {
    compute_type = CUBLAS_COMPUTE_64F;
  }
  return compute_type;
}
#endif

int DenseGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto dout_shape = inputs[kIndex2]->GetShapeVector();
  auto dx_shape = outputs[kIndex0]->GetShapeVector();

  auto dims = dout_shape.size();
  if (dims < kDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of dout cannot be less than 2, but got " << dims;
  }

  m_ = SizeToInt(dx_shape[dims - kDimOffset2]);
  for (size_t i = 0; i < dims - kDimOffset2; i++) {
    m_ *= SizeToInt(dx_shape[i]);
  }
  n_ = SizeToInt(dx_shape[dims - 1]);
  k_ = SizeToInt(dout_shape[dims - 1]);

  lda_ = k_;
  ldb_ = n_;
  ldc_ = n_;

#if CUDA_VERSION >= 11000
  compute_type_ = GetComputeType();
  if (compute_type_ == CUBLAS_COMPUTE_32I) {
    constexpr int bytes = 4;
    if (lda_ % bytes != 0 || ldb_ % bytes != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the lda and ldb must be multiples of 4 when"
                        << "the compute_type_ is CUBLAS_COMPUTE_32I. But got lda:" << lda_ << ", got ldb:" << ldb_;
    }
  }

  auto math_mode = static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetMathMode(handle_, math_mode), "cublasSetMathMode failed.");
#else
  compute_type_ = (dtype_x_ == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F;
  if (dtype_x_ == CUDA_C_32F || dtype_x_ == CUDA_C_64F) {
    compute_type_ = dtype_x_;
  }
#endif

  if (has_bias_) {
    cudnnDataType_t cudnn_data_type = GetCudnnDataType(TypeIdLabel(inputs[kIndex2]->GetDtype()));

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type, m_, k_, 1, 1),
      "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(db_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type, 1, k_, 1, 1),
      "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(op_desc_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                     CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
    InitSizeLists();
  }

  return KRET_OK;
}

template <typename T, typename S>
bool DenseGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto x_addr = GetDeviceAddress<T>(inputs, 0);
  auto w_addr = GetDeviceAddress<T>(inputs, 1);
  auto dout_addr = GetDeviceAddress<T>(inputs, 2);
  auto dx_addr = GetDeviceAddress<T>(outputs, 0);
  auto dw_addr = GetDeviceAddress<T>(outputs, 1);

  S alpha = static_cast<S>(1.0f);
  S beta = static_cast<S>(0.0f);

  if (has_bias_) {
    auto db_addr = GetDeviceAddress<T>(outputs, 2);
    if (m_ == 1) {
      cudaMemcpyAsync(db_addr, dout_addr, k_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      T *indices_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex0);
      T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex1);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, op_desc_, indices_addr, workspace_size_list_[kIndex0], workspace_addr,
                          workspace_size_list_[kIndex1], &alpha, dy_desc_, dout_addr, &beta, db_desc_, db_addr),
        "cudnnReduceTensor failed");
    }
  }

  // dx = dout @ w
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, m_, k_, &alpha, w_addr, dtype_w_, ldb_, dout_addr, dtype_dout_,
                 lda_, &beta, dx_addr, dtype_dx_, ldc_, compute_type_, algo_),
    "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");

  // dw = dout^T @ x
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_T, n_, k_, m_, &alpha, x_addr, dtype_x_, ldb_, dout_addr, dtype_dout_,
                 lda_, &beta, dw_addr, dtype_dw_, ldc_, compute_type_, algo_),
    "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");

  return true;
}

void DenseGradGpuKernelMod::InitResource() {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&db_desc_), "cudnnCreateTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&op_desc_),
                                      "cudnnCreateOpTensorDescriptor failed");
}

void DenseGradGpuKernelMod::InitSizeLists() {
  size_t indices_size, workspace_size;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetReductionIndicesSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &indices_size),
    "cudnnGetReductionIndicesSize failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetReductionWorkspaceSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &workspace_size),
    "cudnnGetReductionWorkspaceSize failed");
  workspace_size_list_.clear();
  workspace_size_list_.push_back(indices_size);
  workspace_size_list_.push_back(workspace_size);
}

std::vector<std::pair<KernelAttr, DenseGradGpuKernelMod::DenseGradFunc>> DenseGradGpuKernelMod::kernel_attr_vec_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &DenseGradGpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &DenseGradGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &DenseGradGpuKernelMod::LaunchKernel<half, float>}};

std::vector<KernelAttr> DenseGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr_vec_.begin(), kernel_attr_vec_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DenseGradFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, DenseGrad,
                                 []() { return std::make_shared<DenseGradGpuKernelMod>(kDenseGradOpName); });
}  // namespace kernel
}  // namespace mindspore
