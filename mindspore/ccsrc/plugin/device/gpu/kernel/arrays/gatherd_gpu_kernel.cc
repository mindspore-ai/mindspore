/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/arrays/gatherd_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
bool GatherDGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  auto input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->device_ptr());
  auto index_addr = reinterpret_cast<S *>(inputs.at(kIndex2)->device_ptr());
  auto output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->device_ptr());
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  auto status = GatherD(input_addr, index_addr, output_addr, dims_[0], dims_[1], dims_[kIndex2], dims_[kIndex3],
                        cuda_stream, GET_CTX_DEVICE_ID);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define REG_INDEX(DT1, DT2, T1, T2)                      \
  {                                                      \
    KernelAttr()                                         \
      .AddInputAttr(DT1)                                 \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddInputAttr(DT2)                                 \
      .AddOutputAttr(DT1),                               \
      &GatherDGpuKernelMod::LaunchKernel<T1, T2>         \
  }

#define GATHER_D_GPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

std::vector<std::pair<KernelAttr, GatherDGpuKernelMod::GatherFwdFunc>> GatherDGpuKernelMod::func_list_ = {
  GATHER_D_GPU_REGISTER(kNumberTypeComplex64, Complex<float>),
  GATHER_D_GPU_REGISTER(kNumberTypeComplex128, Complex<double>),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat16, half),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat32, float),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat64, double),
  GATHER_D_GPU_REGISTER(kNumberTypeInt8, uchar),
  GATHER_D_GPU_REGISTER(kNumberTypeInt16, int16_t),
  GATHER_D_GPU_REGISTER(kNumberTypeInt32, int32_t),
  GATHER_D_GPU_REGISTER(kNumberTypeInt64, int64_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt8, uint8_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt16, uint16_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt32, uint32_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt64, uint64_t),
  GATHER_D_GPU_REGISTER(kNumberTypeBool, bool)};

bool GatherDGpuKernelMod::SetDimParam(int64_t dim_value) {
  int64_t x_rank = SizeToLong(input_shapes_.size());
  if (dim_value < 0) {
    dim_value += x_rank;
  }

  size_t dim_before_axis = 1;
  for (size_t i = 0; i < LongToSize(dim_value); i++) {
    dim_before_axis *= output_shapes_[i];
  }
  size_t dim_at_axis_input = input_shapes_[LongToSize(dim_value)];
  size_t dim_at_axis_output = output_shapes_[LongToSize(dim_value)];
  size_t dim_after_axis = 1;
  for (size_t i = LongToSize(dim_value) + 1; i < output_shapes_.size(); i++) {
    dim_after_axis *= output_shapes_[i];
  }

  dims_[0] = dim_before_axis;
  dims_[1] = dim_at_axis_input;
  dims_[kIndex2] = dim_at_axis_output;
  dims_[kIndex3] = dim_after_axis;
  return true;
}

bool GatherDGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GatherDGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto input_shapes = inputs[0]->GetShapeVector();
  auto index_shapes = inputs[kIndex2]->GetShapeVector();
  auto output_shapes = outputs[0]->GetShapeVector();

  input_shapes_.clear();
  index_shapes_.clear();
  output_shapes_.clear();
  std::transform(input_shapes.cbegin(), input_shapes.cend(), std::back_inserter(input_shapes_), LongToSize);
  std::transform(index_shapes.cbegin(), index_shapes.cend(), std::back_inserter(index_shapes_), LongToSize);
  std::transform(output_shapes.cbegin(), output_shapes.cend(), std::back_inserter(output_shapes_), LongToSize);

  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(index_shapes_, kernel_name_, "input_indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  int64_t dim_value = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (!SetDimParam(dim_value)) {
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

std::vector<KernelAttr> GatherDGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherFwdFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GatherD, GatherDGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
