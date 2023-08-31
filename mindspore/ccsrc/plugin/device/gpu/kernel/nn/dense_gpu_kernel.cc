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

#include "plugin/device/gpu/kernel/nn/dense_gpu_kernel.h"
#include <map>
#include <algorithm>
#include <utility>
#include <memory>
#include "ops/nn_op_name.h"
#include "ops/dense.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/math/matmul/matmul_wrapper.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_v2_impl.cuh"

namespace mindspore {
namespace kernel {
bool DenseGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'Dense', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, DenseGpuKernelMod::DenseFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;

  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  auto dtype_str = TypeIdLabel(inputs[kIndex0]->GetDtype());
  const std::vector<std::string> need_cast_dtypes = {"Int8", "Int16", "Int32", "Int64", "UInt8"};
  auto it = std::find(need_cast_dtypes.begin(), need_cast_dtypes.end(), dtype_str);
  need_cast_ = it != need_cast_dtypes.end();
  if (need_cast_) {
    dtype_a_ = CUDA_R_32F;
    dtype_b_ = CUDA_R_32F;
    dtype_c_ = CUDA_R_32F;
  } else {
    dtype_a_ = GetCudaDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
    dtype_b_ = GetCudaDataType(TypeIdLabel(inputs[kIndex1]->GetDtype()));
    dtype_c_ = GetCudaDataType(TypeIdLabel(outputs[kIndex0]->GetDtype()));
  }

  if (dtype_a_ == CUDA_R_16F) {
    algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  has_bias_ = GetValue<bool>(base_operator->GetAttr("has_bias"));
  compute_type_ = GetComputeType(dtype_a_);

  return true;
}

int DenseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto out_shape = outputs[kIndex0]->GetShapeVector();
  auto x_shape = inputs[kIndex0]->GetShapeVector();

  is_empty_tensor_ = std::any_of(x_shape.begin(), x_shape.end(), [](const int64_t shape) { return shape == 0; });
  auto dims = out_shape.size();
  if (dims == 0) {
    m_ = n_ = 1;
    k_ = x_shape[0];
  } else {
    m_ = out_shape[dims - kDimOffset2];
    for (size_t i = 0; i < dims - kDimOffset2; i++) {
      m_ *= out_shape[i];
    }
    n_ = out_shape.back();
    k_ = x_shape.back();
  }
  if (has_bias_) {
    auto b_shape = inputs[kIndex2]->GetShapeVector();
    b_size_ = b_shape.size() == 0 ? 1 : n_;
  }
  ResetResource();
  return KRET_OK;
}

void DenseGpuKernelMod::ResetSize() {
  lda_ = SizeToInt(k_);
  ldb_ = SizeToInt(k_);
  ldc_ = SizeToInt(n_);
  x_size_ = m_ * k_;
  w_size_ = n_ * k_;
  out_size_ = m_ * n_;
}

void DenseGpuKernelMod::ResetWorkspace() {
  workspace_size_list_.push_back(sizeof(float) * x_size_);
  workspace_size_list_.push_back(sizeof(float) * w_size_);
  workspace_size_list_.push_back(sizeof(float) * out_size_);
  if (has_bias_) {
    workspace_size_list_.push_back(sizeof(float) * b_size_);
  }
}

void DenseGpuKernelMod::ResetResource() {
  ResetSize();
  if (need_cast_) {
    ResetWorkspace();
  }
}

template <typename T>
cudaError_t DenseGpuKernelMod::FillBias(T *src, T *dst, cudaStream_t stream) {
  cudaError_t status;
  if (b_size_ == 1) {
    status = FillV2(out_size_, src, dst, device_id_, stream);
  } else {
    status = Fill(m_, n_, src, dst, stream);
  }
  return status;
}

template <typename T, typename S>
bool DenseGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto x = GetDeviceAddress<T>(inputs, kIndex0);
  auto w = GetDeviceAddress<T>(inputs, kIndex1);
  auto out = GetDeviceAddress<T>(outputs, kIndex0);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cudaError_t status;

  if (is_empty_tensor_) {
    if (has_bias_) {
      auto b = GetDeviceAddress<T>(inputs, kIndex2);
      status = FillBias(b, out, stream);
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
    return true;
  }

  if (need_cast_) {
    auto cast_x = GetPossiblyNullDeviceAddress<float>(workspace, kIndex0);
    auto cast_w = GetPossiblyNullDeviceAddress<float>(workspace, kIndex1);
    auto cast_out = GetPossiblyNullDeviceAddress<float>(workspace, kIndex2);

    status = Cast(x_size_, x, cast_x, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = Cast(w_size_, w, cast_w, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);

    float alpha = 1.0f;
    float beta = 0.0f;

    if (has_bias_) {
      auto b = GetDeviceAddress<T>(inputs, kIndex2);
      auto cast_b = GetPossiblyNullDeviceAddress<float>(workspace, kIndex3);
      status = Cast(b_size_, b, cast_b, stream);
      CHECK_CUDA_STATUS(status, kernel_name_);
      status = FillBias(cast_b, cast_out, stream);
      CHECK_CUDA_STATUS(status, kernel_name_);
      beta = 1.0f;
    }

    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha, cast_w,
                   dtype_b_, ldb_, cast_x, dtype_a_, lda_, &beta, cast_out, dtype_c_, ldc_, compute_type_, algo_),
      "cublasGemmEx failed.");

    status = Cast(out_size_, cast_out, out, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  S alpha = static_cast<S>(1.0f);
  S beta = static_cast<S>(0.0f);

  if (has_bias_) {
    auto b = GetDeviceAddress<T>(inputs, kIndex2);
    status = FillBias(b, out, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
    beta = static_cast<S>(1.0f);
  }

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha, w, dtype_b_,
                 ldb_, x, dtype_a_, lda_, &beta, out, dtype_c_, ldc_, compute_type_, algo_),
    "cublasGemmEx failed.");
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;

std::map<std::string, std::vector<std::pair<KernelAttr, DenseGpuKernelMod::DenseFunc>>>
  DenseGpuKernelMod::kernel_attr_map_ = {
    {kDenseOpName,
     {{KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &DenseGpuKernelMod::LaunchKernel<half, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &DenseGpuKernelMod::LaunchKernel<float, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &DenseGpuKernelMod::LaunchKernel<double, double>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &DenseGpuKernelMod::LaunchKernel<Complex<float>, Complex<float>>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &DenseGpuKernelMod::LaunchKernel<Complex<double>, Complex<double>>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &DenseGpuKernelMod::LaunchKernel<uint8_t, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &DenseGpuKernelMod::LaunchKernel<int8_t, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &DenseGpuKernelMod::LaunchKernel<int16_t, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &DenseGpuKernelMod::LaunchKernel<int32_t, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &DenseGpuKernelMod::LaunchKernel<int64_t, float>}}}};

std::vector<KernelAttr> DenseGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DenseFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Dense,
                                 []() { return std::make_shared<DenseGpuKernelMod>(kDenseOpName); });
}  // namespace kernel
}  // namespace mindspore
