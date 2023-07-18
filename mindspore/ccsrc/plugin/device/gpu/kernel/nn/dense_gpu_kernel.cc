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
  dtype_a_ = GetCudaDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
  dtype_b_ = GetCudaDataType(TypeIdLabel(inputs[kIndex1]->GetDtype()));
  dtype_c_ = GetCudaDataType(TypeIdLabel(outputs[kIndex0]->GetDtype()));

  if (dtype_a_ == CUDA_R_16F) {
    algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  has_bias_ = GetValue<bool>(base_operator->GetAttr("has_bias"));

  return true;
}

#if CUDA_VERSION >= 11000
cublasComputeType_t DenseGpuKernelMod::GetComputeType() {
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

int DenseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto input1_shape = inputs[kIndex0]->GetShapeVector();

  auto dims = output_shape.size();
  if (dims < kDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 2, but got "
                      << dims;
  }

  m_ = output_shape[dims - kDimOffset2];
  for (size_t i = 0; i < dims - kDimOffset2; i++) {
    m_ *= output_shape[i];
  }
  n_ = output_shape[dims - 1];

  if (input1_shape.size() > (dims - 1)) {
    k_ = input1_shape[dims - 1];
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', init k_ via input1_shape failed.";
  }
  lda_ = SizeToInt(k_);
  ldb_ = SizeToInt(k_);
  ldc_ = SizeToInt(n_);

#if CUDA_VERSION >= 11000
  compute_type_ = GetComputeType();
  if (compute_type_ == CUBLAS_COMPUTE_32I) {
    constexpr size_t bytes = 4;
    if (lda_ % bytes != 0 || ldb_ % bytes != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "' the lda and ldb must be multiples of 4 when the compute_type_ is CUBLAS_COMPUTE_32I."
                           "But got lda:"
                        << lda_ << ", got ldb:" << ldb_;
    }
  }

  auto math_mode = static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetMathMode(handle_, math_mode), "cublasSetMathMode failed.");
#else
  compute_type_ = (dtype_a_ == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F;
  if (dtype_a_ == CUDA_C_32F || dtype_a_ == CUDA_C_64F) {
    compute_type_ = dtype_a_;
  }
#endif

  return KRET_OK;
}

template <typename T, typename S>
bool DenseGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input1_addr = GetDeviceAddress<T>(inputs, 0);
  auto input2_addr = GetDeviceAddress<T>(inputs, 1);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);

  S alpha = static_cast<S>(1.0f);
  S beta = static_cast<S>(0.0f);

  if (has_bias_) {
    auto input3_addr = GetDeviceAddress<T>(inputs, 2);
    Fill(m_, n_, input3_addr, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    beta = static_cast<S>(1.0f);
  }

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha, input2_addr,
                 dtype_b_, ldb_, input1_addr, dtype_a_, lda_, &beta, output_addr, dtype_c_, ldc_, compute_type_, algo_),
    "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");
  return true;
}

std::map<std::string, std::vector<std::pair<KernelAttr, DenseGpuKernelMod::DenseFunc>>>
  DenseGpuKernelMod::kernel_attr_map_ = {
    {kDenseOpName,
     {{KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &DenseGpuKernelMod::LaunchKernel<double, double>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &DenseGpuKernelMod::LaunchKernel<float, float>},
      {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &DenseGpuKernelMod::LaunchKernel<half, float>}}}};

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
