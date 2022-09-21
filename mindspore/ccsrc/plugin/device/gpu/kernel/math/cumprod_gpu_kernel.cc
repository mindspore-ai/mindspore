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

#include "plugin/device/gpu/kernel/math/cumprod_gpu_kernel.h"
#include "mindspore/core/ops/cumprod.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumProdInputsNum = 1;
constexpr size_t kCumProdDynamicInputsNum = 2;
constexpr size_t kCumProdOutputsNum = 1;
constexpr size_t kDimSize0 = 0;
constexpr size_t kDimSize1 = 1;
constexpr size_t kDimSize2 = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

template <typename T>
bool CumProdGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumProdOutputsNum, kernel_name_);
  auto input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto ws_addr = GetDeviceAddress<T>(workspace, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input_addr, output_addr, ws_addr)) {
    return false;
  }
  if (is_dynamic_shape_) {
    auto axis_addr = GetDeviceAddress<T>(inputs, kIndex1);
    if (axis_addr == nullptr) {
      return false;
    }
    int64_t axis_tmp;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&axis_tmp, axis_addr, inputs[kIndex1]->size, cudaMemcpyDeviceToHost),
                                       "For '" << kernel_name_ << "', cudaMemcpy input 'axis' device to host failed.");
    axis_ = static_cast<int>(axis_tmp);
    Reshape();
  }
  CumProd(input_addr, output_addr, ws_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_,
          exclusive_, reverse_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool CumProdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CumProd>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();
  exclusive_ = kernel_ptr->GetExclusive();
  reverse_ = kernel_ptr->GetReverse();

  auto input_num = inputs.size();
  if (input_num == kCumProdInputsNum) {
    is_dynamic_shape_ = false;
  } else if (input_num == kCumProdDynamicInputsNum) {
    is_dynamic_shape_ = true;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int CumProdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  auto shape_signed = inputs[kIndex0]->GetShapeVector();
  shape_ = Convert2SizeTClipNeg(shape_signed);
  is_null_input_ = CHECK_SHAPE_NULL(shape_, kernel_name_, "input");
  if (is_null_input_) {
    workspace_size_list_.push_back(input_size_0_);
    return KRET_OK;
  }

  int input_dim_length = SizeToInt(shape_.size());
  if (!is_dynamic_shape_) {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::CumProd>(base_operator);
    MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
    axis_ = static_cast<int>(kernel_ptr->GetAxis());
    if (axis_ >= input_dim_length) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << ", 'axis' should be less than the length of 'input' dimension, but got 'axis': " << axis_
                    << " and the length of 'input' dimension: " << input_dim_length;
      return KRET_RESIZE_FAILED;
    }
    Reshape();
  }
  workspace_size_list_.push_back(input_size_list_.at(kIndex0));
  return KRET_OK;
}

void CumProdGpuKernelMod::Reshape() {
  while (axis_ < 0) {
    axis_ += SizeToInt(shape_.size());
  }
  dims_[kIndex0] = 1;
  dims_[kIndex1] = shape_[IntToSize(axis_)];
  dims_[kIndex2] = 1;
  for (size_t i = 0; i < IntToSize(axis_); i++) {
    dims_[kIndex0] *= shape_[i];
  }
  for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
    dims_[kIndex2] *= shape_[i];
  }
  stride_ = dims_[kIndex1] * dims_[kIndex2];
  stride2_ = dims_[kIndex2];
}

using cumProdPair = std::pair<KernelAttr, CumProdGpuKernelMod::KernelRunFunc>;
const std::vector<cumProdPair> &CumProdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CumProdGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CumProdGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CumProdGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &CumProdGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CumProdGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CumProdGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &CumProdGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &CumProdGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &CumProdGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CumProdGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CumProdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CumProdGpuKernelMod::LaunchKernel<double>},
    // Dynamic shape related.
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &CumProdGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &CumProdGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &CumProdGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CumProdGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &CumProdGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &CumProdGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &CumProdGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &CumProdGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &CumProdGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &CumProdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &CumProdGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CumProd, CumProdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
