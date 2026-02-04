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

#include "kernel/gpu/cuda/math/diagonal_gpu_kernel.h"
#include <functional>
#include <string>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDiagonalInputsNum = 4;
constexpr size_t kDiagonalOutputsNum = 1;
constexpr int64_t N2 = 2;
template <typename T>
T mul_sum(std::vector<T> v1, std::vector<T> v2) {
  T output = 0;
  for (unsigned int i = 0; i < v1.size(); i++) {
    output += v1[i] * v2[i];
  }
  return output;
}

template <typename T>
std::vector<T> construct_stride(std::vector<T> t_shape) {
  std::vector<T> t_stride(t_shape.size(), 1);
  int initial = 1;
  for (unsigned int i = t_shape.size(); i > 0; i--) {
    t_stride[i - 1] = initial;
    initial = initial * t_shape[i - 1];
  }
  return t_stride;
}
}  // namespace

bool DiagonalGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For 'Diagonal', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Diagonal', the data type of input must be float32 or double, but got: " << kernel_attr
                  << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int DiagonalGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagonalInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDiagonalOutputsNum, kernel_name_);
  offset_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  dim1_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  dim2_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape = inputs[0]->GetShapeVector();
  output_shape = outputs[0]->GetShapeVector();
  int64_t input_size = input_shape.size();
  if (input_size < N2) {
    MS_LOG(ERROR) << "For 'Diagonal', input must be at least 2-dimensional, but got : " << input_size << ".";
    return false;
  }
  if (dim1_ > input_size - 1 || dim1_ < -input_size) {
    MS_LOG(ERROR) << "For 'Diagonal', dim1 should be in range of [" << -input_size << "," << (input_size - 1)
                  << "], but got : " << dim1_ << ".";
    return false;
  }
  if (dim2_ > input_size - 1 || dim2_ < -input_size) {
    MS_LOG(ERROR) << "For 'Diagonal', dim2 should be in range of [" << -input_size << "," << (input_size - 1)
                  << "], but got : " << dim2_ << ".";
    return false;
  }
  dim1_ = (dim1_ < 0) ? dim1_ + input_size : dim1_;
  dim2_ = (dim2_ < 0) ? dim2_ + input_size : dim2_;
  if (dim1_ == dim2_) {
    MS_LOG(ERROR) << "For 'Diagonal', dim1 and dim2 cannot be identical, but got : dim1 =" << dim1_
                  << " and dim2 = " << dim2_ << ".";
  }

  return KRET_OK;
}

template <typename T>
bool DiagonalGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagonalInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDiagonalOutputsNum, kernel_name_);

  const T *input = GetDeviceAddress<T>(inputs, 0);
  MS_EXCEPTION_IF_NULL(input);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto status = CalDiagonal(input, offset_, dim1_, dim2_, input_shape, output_shape, output, device_id_,
                            reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, DiagonalGpuKernelMod::DiagonalLaunchFunc>> DiagonalGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DiagonalGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &DiagonalGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &DiagonalGpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &DiagonalGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &DiagonalGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &DiagonalGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &DiagonalGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &DiagonalGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &DiagonalGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &DiagonalGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &DiagonalGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &DiagonalGpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> DiagonalGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DiagonalLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Diagonal, DiagonalGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
