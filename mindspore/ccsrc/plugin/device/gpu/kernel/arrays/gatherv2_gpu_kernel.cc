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

#include "plugin/device/gpu/kernel/arrays/gatherv2_gpu_kernel.h"
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
const size_t kInputNum = 3;
bool GatherV2FwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  indices_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  axis_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  axis_type_ = inputs.at(kIndex2)->GetDtype();
  return true;
}

int GatherV2FwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
  }
  if (TryGetIntValue(inputs, kIndex2, kernel_name_, &axis_)) {
    is_get_axis_ = true;
  }
  ResetResource();
  input_shapes_ = inputs[kIndexZero]->GetShapeVector();
  indices_shapes_ = inputs[kIndexOne]->GetShapeVector();
  output_shapes_ = outputs[kIndexZero]->GetShapeVector();
  if (IsDynamic(input_shapes_) || IsDynamic(indices_shapes_) || IsDynamic(output_shapes_)) {
    return KRET_UNKNOWN_SHAPE;
  }
  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shapes_, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  int dims = SizeToInt(input_shapes_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                      << "), but got " << axis_;
  }
  Reshape();
  InitSizeLists();
  return KRET_OK;
}

std::vector<KernelAttr> GatherV2FwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherV2Func> &pair) { return pair.first; });
  return support_list;
}

template <typename T, typename S, typename G>
bool GatherV2FwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (!is_get_axis_) {
    axis_ = GetDimValue<int64_t>(inputs, kIndex2, kernel_name_, axis_type_);
  }
  auto input_dim1 = input_shapes_[IntToSize(axis_)];

  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(indices_addr);
  GatherV2(input_addr, indices_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
           LongToSize(input_dim1), reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

#define GATHER_GPU_REG(MS_T, MS_S, MS_A, T, S, A)                                            \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_A).AddOutputAttr(MS_T), \
    &GatherV2FwdGpuKernelMod::LaunchKernel<T, S, A>

#define GATHER_GPU_INDEX_REG(MS_T, T)                                                  \
  {GATHER_GPU_REG(MS_T, kNumberTypeInt32, kNumberTypeInt32, T, int32_t, int32_t)},     \
    {GATHER_GPU_REG(MS_T, kNumberTypeInt64, kNumberTypeInt64, T, int64_t, int64_t)},   \
    {GATHER_GPU_REG(MS_T, kNumberTypeInt32, kNumberTypeInt64, T, int32_t, int64_t)}, { \
    GATHER_GPU_REG(MS_T, kNumberTypeInt64, kNumberTypeInt32, T, int64_t, int32_t)      \
  }

std::vector<std::pair<KernelAttr, GatherV2FwdGpuKernelMod::GatherV2Func>> GatherV2FwdGpuKernelMod::func_list_ = {{
  GATHER_GPU_INDEX_REG(kNumberTypeComplex64, mindspore::utils::Complex<float>),
  GATHER_GPU_INDEX_REG(kNumberTypeComplex128, mindspore::utils::Complex<double>),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat64, double),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat32, float),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat16, half),
  GATHER_GPU_INDEX_REG(kNumberTypeInt64, int64_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt32, int32_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt16, int16_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt8, int8_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt64, uint64_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt32, uint32_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt16, uint16_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt8, uint8_t),
  GATHER_GPU_INDEX_REG(kNumberTypeBool, bool),
}};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Gather,
                                 []() { return std::make_shared<GatherV2FwdGpuKernelMod>(kGather); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseGatherV2,
                                 []() { return std::make_shared<GatherV2FwdGpuKernelMod>(kSparseGatherV2); });
}  // namespace kernel
}  // namespace mindspore
