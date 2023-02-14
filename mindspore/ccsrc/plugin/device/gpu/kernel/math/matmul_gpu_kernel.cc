/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/matmul_gpu_kernel.h"
#include <map>
#include <algorithm>
#include <utility>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "mindspore/core/ops/mat_mul.h"
#include "mindspore/core/ops/batch_matmul.h"

namespace mindspore {
namespace kernel {
namespace {
inline bool IsComplex(cudaDataType_t type) { return type == CUDA_C_32F || type == CUDA_C_64F; }
}  // namespace

bool MatMulGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatMul>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast" << kernel_name_ << " failed!";
    return false;
  }

  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'MatMul', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, MatMulGpuKernelMod::MatMulFunc>>>(
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
  dtype_a_ = GetCudaDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
  dtype_b_ = GetCudaDataType(TypeIdLabel(inputs[kIndex1]->GetDtype()));
  dtype_c_ = GetCudaDataType(TypeIdLabel(outputs[kIndex0]->GetDtype()));

  if (dtype_a_ != dtype_b_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the types of inputs are not the same.";
  }
  if (dtype_a_ == CUDA_R_16F && dtype_b_ == CUDA_R_16F && dtype_c_ == CUDA_R_16F) {
    MS_LOG(INFO) << "input and output type is float16, allow to use Tensor Core operations if possible";
    algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  transpose_x1_ = kernel_ptr->get_transpose_a() ? CUBLAS_OP_T : CUBLAS_OP_N;
  transpose_x2_ = kernel_ptr->get_transpose_b() ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (transpose_x1_ != CUBLAS_OP_N && IsComplex(dtype_a_)) {
    transpose_x1_ = CUBLAS_OP_C;
  }
  if (transpose_x2_ != CUBLAS_OP_N && IsComplex(dtype_b_)) {
    transpose_x2_ = CUBLAS_OP_C;
  }
  if (kernel_name_ == kFusedMatMulBiasAddName) {
    is_fused_matmul_biasadd_ = true;
  }
  return true;
}

int MatMulGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto output_shape_signed = outputs[kIndex0]->GetShapeVector();
  auto input1_shape_signed = inputs[kIndex0]->GetShapeVector();
  auto output_shape = Convert2SizeTClipNeg(output_shape_signed);
  auto input1_shape = Convert2SizeTClipNeg(input1_shape_signed);

  auto dims = output_shape.size();
  if (dims < kDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 2, but got "
                      << dims;
  }

  m_ = output_shape[dims - kDimOffset2];
  n_ = output_shape[dims - 1];
  batch_ = 1;
  for (size_t i = 0; i < dims - kDimOffset2; i++) {
    batch_ *= output_shape[i];
  }

  if (transpose_x1_ != CUBLAS_OP_N && input1_shape.size() > (dims - kDimOffset2)) {
    k_ = input1_shape[dims - kDimOffset2];
  } else if (input1_shape.size() > (dims - 1)) {
    k_ = input1_shape[dims - 1];
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', init k_ via input1_shape failed.";
  }

  return KRET_OK;
}

#if CUDA_VERSION >= 11000
cublasComputeType_t MatMulGpuKernelMod::GetComputeType() {
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (dtype_a_ == CUDA_R_16F && dtype_c_ == CUDA_R_16F) {
    compute_type = CUBLAS_COMPUTE_32F;
  } else if (dtype_a_ == CUDA_R_8I && dtype_c_ == CUDA_R_32I) {
    compute_type = CUBLAS_COMPUTE_32I;
  } else if (dtype_a_ == CUDA_R_16F || dtype_a_ == CUDA_R_32F || (dtype_a_ == CUDA_R_8I && dtype_c_ == CUDA_R_32F)) {
    compute_type = CUBLAS_COMPUTE_32F;
  } else if ((dtype_a_ == CUDA_R_32F && dtype_b_ == CUDA_R_32F) || (dtype_a_ == CUDA_C_32F && dtype_b_ == CUDA_C_32F)) {
    compute_type = CUBLAS_COMPUTE_32F;
  } else if ((dtype_a_ == CUDA_R_64F && dtype_b_ == CUDA_R_64F) || (dtype_a_ == CUDA_C_64F && dtype_b_ == CUDA_C_64F)) {
    compute_type = CUBLAS_COMPUTE_64F;
  }
  return compute_type;
}
#endif

template <typename T, typename S>
bool MatMulGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input1_addr = GetDeviceAddress<T>(inputs, 0);
  auto input2_addr = GetDeviceAddress<T>(inputs, 1);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);

  // The alpha & beta should be float if the inputs are half, else should keep the same type with inputs.
  // The kernel registration is responsible for types consistency.
  S alpha = static_cast<S>(1.0f);
  S beta = static_cast<S>(0.0f);
  const int lda = (transpose_x1_ != CUBLAS_OP_N) ? SizeToInt(m_) : SizeToInt(k_);
  const int ldb = (transpose_x2_ != CUBLAS_OP_N) ? SizeToInt(k_) : SizeToInt(n_);
  const int ldc = n_;

  try {
    if (is_fused_matmul_biasadd_) {
      auto input3_addr = GetDeviceAddress<T>(inputs, 2);
      Fill(m_, n_, input3_addr, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      beta = static_cast<S>(1.0f);
    }

#if CUDA_VERSION >= 11000
    cublasComputeType_t compute_type = GetComputeType();
    if (compute_type == CUBLAS_COMPUTE_32I) {
      constexpr size_t bytes = 4;
      if (lda % bytes != 0 || ldb % bytes != 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "' the lda and ldb must be multiples of 4 when the compute_type is CUBLAS_COMPUTE_32I."
                             "But got lda:"
                          << lda << ", got ldb:" << ldb;
      }
    }
#else
    cudaDataType_t compute_type = (dtype_a_ == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F;
    if (dtype_a_ == CUDA_C_32F || dtype_a_ == CUDA_C_64F) {
      compute_type = dtype_a_;
    }
#endif

    // Use cublasGemmEx to get high performance when batch_ is 1
    if (batch_ == 1) {
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasGemmEx(handle_, transpose_x2_, transpose_x1_, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha,
                     input2_addr, dtype_b_, ldb, input1_addr, dtype_a_, lda, &beta, output_addr, dtype_c_, ldc,
                     compute_type, algo_),
        "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");
    } else {
      auto stride_a = SizeToLong(m_ * k_);
      auto stride_b = SizeToLong(k_ * n_);
      auto stride_c = SizeToLong(m_ * n_);
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasGemmStridedBatchedEx(handle_, transpose_x2_, transpose_x1_, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_),
                                   &alpha, input2_addr, dtype_b_, ldb, stride_b, input1_addr, dtype_a_, lda, stride_a,
                                   &beta, output_addr, dtype_c_, ldc, stride_c, batch_, compute_type, algo_),
        "cublasGemmStridedBatchedEx failed. Possible reasons: the GPU is occupied by other processes.");
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', encountered an exception: " << e.what() << " when invoke "
                      << "cublas " << (batch_ == 1 ? "cublasGemmEx" : "cublasGemmStridedBatchedEx");
  }
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::map<std::string, std::vector<std::pair<KernelAttr, MatMulGpuKernelMod::MatMulFunc>>>
  MatMulGpuKernelMod::kernel_attr_map_ = {
    {kMatMulOpName,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &MatMulGpuKernelMod::LaunchKernel<Complex<float>, Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &MatMulGpuKernelMod::LaunchKernel<double, double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &MatMulGpuKernelMod::LaunchKernel<float, float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &MatMulGpuKernelMod::LaunchKernel<half, float>}}},
    {kBatchMatMulOpName,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &MatMulGpuKernelMod::LaunchKernel<double, double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &MatMulGpuKernelMod::LaunchKernel<float, float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &MatMulGpuKernelMod::LaunchKernel<half, float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
       &MatMulGpuKernelMod::LaunchKernel<int8_t, int32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &MatMulGpuKernelMod::LaunchKernel<Complex<float>, Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &MatMulGpuKernelMod::LaunchKernel<Complex<double>, Complex<double>>}}},
    {kFusedMatMulBiasAddName,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &MatMulGpuKernelMod::LaunchKernel<double, double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &MatMulGpuKernelMod::LaunchKernel<float, float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &MatMulGpuKernelMod::LaunchKernel<half, float>}}}};

std::vector<KernelAttr> MatMulGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'MatMul', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, MatMulGpuKernelMod::MatMulFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatMulFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MatMul,
                                 []() { return std::make_shared<MatMulGpuKernelMod>(kMatMulOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BatchMatMul,
                                 []() { return std::make_shared<MatMulGpuKernelMod>(kBatchMatMulOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, FusedMatMulBiasAdd,
                                 []() { return std::make_shared<MatMulGpuKernelMod>(kFusedMatMulBiasAddName); });
}  // namespace kernel
}  // namespace mindspore
