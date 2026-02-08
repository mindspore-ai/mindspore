/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/one_hot_gpu_kernel.h"
#include <cstdint>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/one_hot_impl.cuh"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr size_t kOneHotInputsNum = 5;
constexpr size_t kOneHotOutputsNum = 1;
#define REG_ONE_HOT_FIVE_INPUT(IndicesEType, IndicesImplType, DepthEType, AxisEType, ValueEType, ValueImplType) \
  {                                                                                                             \
    KernelAttr()                                                                                                \
      .AddInputAttr(IndicesEType)                                                                               \
      .AddInputAttr(kObjectTypeNumber, DepthEType)                                                              \
      .AddInputAttr(ValueEType)                                                                                 \
      .AddInputAttr(ValueEType)                                                                                 \
      .AddInputAttr(kObjectTypeNumber, AxisEType)                                                               \
      .AddOutputAttr(ValueEType),                                                                               \
      &OneHotGpuKernelMod::LaunchKernel<ValueImplType, IndicesImplType>                                         \
  }

#define REG_ONE_HOT_GPU_KERNEL(ValueEnumType, ValueImplType)                                                         \
  REG_ONE_HOT_FIVE_INPUT(kNumberTypeInt32, int, kNumberTypeInt64, kNumberTypeInt64, ValueEnumType, ValueImplType),   \
    REG_ONE_HOT_FIVE_INPUT(kNumberTypeInt64, int64_t, kNumberTypeInt64, kNumberTypeInt64, ValueEnumType,             \
                           ValueImplType),                                                                           \
    REG_ONE_HOT_FIVE_INPUT(kNumberTypeInt32, int, kNumberTypeInt32, kNumberTypeInt64, ValueEnumType, ValueImplType), \
    REG_ONE_HOT_FIVE_INPUT(kNumberTypeInt64, int64_t, kNumberTypeInt32, kNumberTypeInt64, ValueEnumType,             \
                           ValueImplType)

bool OneHotGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOneHotInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOneHotOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int OneHotGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto output_shape = LongVecToSizeVec(outputs[kIndex0]->GetShapeVector());
  int64_t axis = inputs[axis_index_]->GetValueWithCheck<int64_t>();

  int64_t input_dims = static_cast<int64_t>(input_shape.size());
  int64_t output_dims = static_cast<int64_t>(output_shape.size());
  if (axis > input_dims || axis >= output_dims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input and output"
                  << ", but got 'axis': " << axis << ", the dimension of input: " << input_dims
                  << ", the dimension of output: " << output_dims;
    return KRET_RESIZE_FAILED;
  }
  const int64_t default_axis = -1;
  left_dim_size_ = 1;
  right_dim_size_ = 1;

  // Compress arbitrary tensor dimensions into three dimensions (left_dims, depth, right_dims).
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto dim_size = input_shape[i];
    if (axis == default_axis || i < IntToSize(axis)) {
      left_dim_size_ *= dim_size;
    }
    if (axis != default_axis && i >= IntToSize(axis)) {
      right_dim_size_ *= dim_size;
    }
  }
  if (axis == default_axis) {
    depth_ = output_shape[output_shape.size() - 1];
  } else {
    depth_ = output_shape[IntToSize(axis)];
  }
  return KRET_OK;
}

template <typename T, typename S>
bool OneHotGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  const size_t on_value_idx = 2;
  const size_t off_value_idx = 3;
  const S *indices = GetDeviceAddress<S>(inputs, 0);
  const T *on_value = GetDeviceAddress<T>(inputs, on_value_idx);
  const T *off_value = GetDeviceAddress<T>(inputs, off_value_idx);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto status = OneHot(indices, depth_, on_value, off_value, left_dim_size_, right_dim_size_, output, device_id_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, OneHotGpuKernelMod::OneHotLaunchFunc>> OneHotGpuKernelMod::func_list_ = {
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeUInt8, uint8_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeUInt16, uint16_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeUInt32, uint32_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeUInt64, uint64_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeInt8, int8_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeInt16, int16_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeInt32, int32_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeInt64, int64_t),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeFloat16, half),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeFloat32, float),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeFloat64, double),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeBool, bool),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeComplex64, utils::Complex<float>),
  REG_ONE_HOT_GPU_KERNEL(kNumberTypeComplex128, utils::Complex<double>)};

std::vector<KernelAttr> OneHotGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, OneHotGpuKernelMod::OneHotLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, OneHot, OneHotGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
