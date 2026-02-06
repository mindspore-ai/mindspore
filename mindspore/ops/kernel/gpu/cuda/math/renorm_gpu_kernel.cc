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

#include "kernel/gpu/cuda/math/renorm_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <set>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "mindspore/ops/infer/renorm.h"
#include "kernel/gpu/cuda_impl/cuda_ops/renorm_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t RenormInputsNum = 1;
constexpr size_t RenormOutputsNum = 1;
bool RenormGpuKernelMod::GetRenormAttr() {
  dim_ = GetValue<int64_t>(primitive_->GetAttr("dim"));
  p_ = GetValue<float>(primitive_->GetAttr("p"));
  if (p_ <= 0.0f) {
    MS_LOG(ERROR) << "For 'Renorm', it's op attribute 'p'" << p_ << "less than or equals to zero is invalid.";
    return false;
  }
  max_norm_ = GetValue<float>(primitive_->GetAttr("maxnorm"));
  if (max_norm_ < 0) {
    MS_LOG(ERROR) << "For 'Renorm', it's op attribute 'maxnorm'" << max_norm_ << "less than zero is invalid.";
    return false;
  }
  return true;
}

template <typename T>
bool RenormGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->device_ptr());
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->device_ptr());
  auto norm_value = reinterpret_cast<float *>(workspace.at(kIndex0)->device_ptr());
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(norm_value, 0, input_shape_[dim_] * sizeof(float), reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'Renorm', it's cudaMemsetAsync failed.");
  auto status = CalRenorm(input, total_size_, inner_size_, axis_size_, p_, norm_value, output, device_id_,
                          reinterpret_cast<cudaStream_t>(cuda_stream_), max_norm_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, RenormGpuKernelMod::RenormFunc>> RenormGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &RenormGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &RenormGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &RenormGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &RenormGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &RenormGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> RenormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RenormFunc> &pair) { return pair.first; });
  return support_list;
}

int RenormGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  InitParams();

  InitWorkSpaceSizeList();
  return KRET_OK;
}

void RenormGpuKernelMod::InitParams() {
  auto shape_size = input_shape_.size();
  MS_EXCEPTION_IF_ZERO("input shape", shape_size);
  if (dim_ >= SizeToLong(shape_size) || dim_ < -SizeToLong(shape_size)) {
    MS_LOG(EXCEPTION) << "For 'Renorm', it's op attribute 'dim' must be in range [" << -SizeToLong(shape_size) << ", "
                      << shape_size << "), but got " << dim_;
  }
  if (dim_ < 0) {
    dim_ += shape_size;
  }
  axis_size_ = 1;
  inner_size_ = 1;
  stride_size_ = 1;
  total_size_ = 1;
  for (size_t i = 0; i < shape_size; ++i) {
    if (SizeToLong(i) == dim_) {
      axis_size_ *= input_shape_[i];
    } else if (SizeToLong(i) < dim_) {
      stride_size_ *= input_shape_[i];
    } else {
      inner_size_ *= input_shape_[i];
    }
    total_size_ *= input_shape_[i];
  }
}

bool RenormGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For 'Renorm', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != RenormInputsNum || outputs.size() != RenormOutputsNum) {
    MS_LOG(ERROR) << "For 'Renorm', input and output tensor number must be 1, but got input tensor number:"
                  << inputs.size() << " and output tensor number:" << outputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Renorm', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return GetRenormAttr();
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Renorm, RenormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
