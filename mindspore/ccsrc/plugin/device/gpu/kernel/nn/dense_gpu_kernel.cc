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

  compute_type_ = GetComputeType(dtype_a_);

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
    auto status = Fill(m_, n_, input3_addr, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    beta = static_cast<S>(1.0f);
  }

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha, input2_addr,
                 dtype_b_, ldb_, input1_addr, dtype_a_, lda_, &beta, output_addr, dtype_c_, ldc_, compute_type_, algo_),
    "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");
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
       &DenseGpuKernelMod::LaunchKernel<Complex<double>, Complex<double>>}}}};

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
