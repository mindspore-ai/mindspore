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

#include "kernel/gpu/cuda/math/cumsum_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumSumInputsNum = 4;
}  // namespace

bool CumSumGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto input_num = inputs.size();
  is_dynamic_shape_ = inputs[kIndex0]->IsDynamicShape();
  if (input_num != kCumSumInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CumSumGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(kIndex0)->GetShapeVector();
  shape_.clear();
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_), LongToSize);
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    return KRET_OK;
  }
  exclusive_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  reverse_ = inputs[kIndex3]->GetValueWithCheck<bool>();
  workspace_size_list_.push_back(inputs[kIndex0]->size());
  return KRET_OK;
}

void CumSumGpuKernelMod::Reshape() {
  int shape_size = SizeToInt(shape_.size());
  if (axis_ < -shape_size || axis_ >= shape_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the `axis` should be in [" << -shape_size << ", " << shape_size
                      << "), but got " << axis_;
  }
  axis_ = (axis_ < 0) ? axis_ + SizeToInt(shape_.size()) : axis_;
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

template <typename T>
bool CumSumGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs) {
  if (is_null_input_) {
    return true;
  }
  auto input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto ws_addr = GetDeviceAddress<T>(workspace, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input_addr, output_addr, ws_addr, cuda_stream_)) {
    return false;
  }
  const auto &axis_addr = inputs.at(kIndex1);
  MS_EXCEPTION_IF_NULL(axis_addr);
  if (axis_addr->size() == sizeof(int)) {
    int axis_tmp;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&axis_tmp, axis_addr->device_ptr(), axis_addr->size(), cudaMemcpyDeviceToHost, cuda_stream_),
      "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
    if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
    }
    axis_ = axis_tmp;
  } else if (inputs.at(kIndex1)->size() == sizeof(int64_t)) {
    int64_t axis_tmp;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&axis_tmp, axis_addr->device_ptr(), axis_addr->size(), cudaMemcpyDeviceToHost, cuda_stream_),
      "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
    if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
    }
    axis_ = static_cast<int>(axis_tmp);
  } else {
    MS_LOG(ERROR) << "The dtype of 'axis' should be int or int64";
    return false;
  }
  Reshape();
  auto status = CumSum(input_addr, output_addr, ws_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_,
                       stride2_, exclusive_, reverse_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, CumSumGpuKernelMod::CumSumLaunchFunc>> CumSumGpuKernelMod::func_list_ = {
  // axis is Scalar
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt8),
   &CumSumGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt16),
   &CumSumGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt32),
   &CumSumGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt64),
   &CumSumGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt8),
   &CumSumGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt16),
   &CumSumGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt32),
   &CumSumGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt64),
   &CumSumGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat16),
   &CumSumGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32),
   &CumSumGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64),
   &CumSumGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex64),
   &CumSumGpuKernelMod::LaunchKernel<utils::Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex128),
   &CumSumGpuKernelMod::LaunchKernel<utils::Complex<double>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt8),
   &CumSumGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt16),
   &CumSumGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt32),
   &CumSumGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeInt64),
   &CumSumGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt8),
   &CumSumGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt16),
   &CumSumGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt32),
   &CumSumGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeUInt64),
   &CumSumGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat16),
   &CumSumGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32),
   &CumSumGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64),
   &CumSumGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex64),
   &CumSumGpuKernelMod::LaunchKernel<utils::Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex128),
   &CumSumGpuKernelMod::LaunchKernel<utils::Complex<double>>}};

std::vector<KernelAttr> CumSumGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumSumGpuKernelMod::CumSumLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CumSum, CumSumGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
