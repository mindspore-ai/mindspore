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

#include "kernel/gpu/cuda/math/cumprod_gpu_kernel.h"
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumProdInputsNum = 4;
constexpr size_t kCumProdOutputsNum = 1;
constexpr size_t kDimSize0 = 0;
constexpr size_t kDimSize1 = 1;
constexpr size_t kDimSize2 = 2;
}  // namespace

template <typename T>
bool CumProdGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumProdOutputsNum, kernel_name_);
  auto input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto ws_addr = GetDeviceAddress<T>(workspace, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input_addr, output_addr, ws_addr)) {
    return false;
  }
  auto axis_addr = GetDeviceAddress<int64_t>(inputs, kIndex1);
  if (axis_addr == nullptr) {
    return false;
  }
  int64_t axis_tmp;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&axis_tmp, axis_addr, inputs[kIndex1]->size(), cudaMemcpyDeviceToHost, cuda_stream_),
    "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
  if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
  }
  axis_ = static_cast<int>(axis_tmp);
  if (axis_ >= input_dim_length_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << ", 'axis' should be less than the length of 'input' dimension, but got 'axis': " << axis_
                  << " and the length of 'input' dimension: " << input_dim_length_;
    return false;
  }
  Reshape();
  auto status = CumProd(input_addr, output_addr, ws_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_,
                        stride2_, exclusive_, reverse_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool CumProdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  is_dynamic_shape_ = inputs[kIndex0]->IsDynamicShape();
  auto input_num = inputs.size();
  if (input_num != kCumProdInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return false;
  }

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int CumProdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  exclusive_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  reverse_ = inputs[kIndex3]->GetValueWithCheck<bool>();
  auto shape_signed = inputs[kIndex0]->GetShapeVector();
  shape_ = Convert2SizeTClipNeg(shape_signed);
  is_null_input_ = CHECK_SHAPE_NULL(shape_, kernel_name_, "input");
  if (is_null_input_) {
    workspace_size_list_.push_back(input_size_0_);
    return KRET_OK;
  }

  input_dim_length_ = SizeToInt(shape_.size());
  size_t input_size = std::accumulate(shape_.begin(), shape_.end(), UnitSizeInBytes(inputs[kIndex0]->dtype_id()),
                                      std::multiplies<size_t>());
  workspace_size_list_.push_back(input_size);
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
    // axis is Scalar
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt8),
     &CumProdGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt16),
     &CumProdGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt32),
     &CumProdGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &CumProdGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeUInt8),
     &CumProdGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeUInt16),
     &CumProdGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeUInt32),
     &CumProdGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeUInt64),
     &CumProdGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat16),
     &CumProdGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat32),
     &CumProdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat64),
     &CumProdGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeComplex64),
     &CumProdGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeComplex128),
     &CumProdGpuKernelMod::LaunchKernel<utils::Complex<double>>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CumProd, CumProdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
