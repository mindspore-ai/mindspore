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

#include "kernel/gpu/cuda/arrays/strided_slice_gpu_kernel.h"
#include <bitset>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S = int64_t>
bool StridedSliceGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (IsEmptyInput(inputs[0]->size())) {
    return true;
  }
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto status = StridedSlice(input_shape_, begin_, strides_, output_shape_, input, output,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool StridedSliceGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int StridedSliceGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto shape_signed = inputs[0]->GetShapeVector();
  input_shape_ = Convert2SizeTClipNeg(shape_signed);
  if (input_shape_.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got " << input_shape_.size();
  }
  begin_ = inputs[kBeginIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  end_ = inputs[kEndIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  strides_ = inputs[kStrideIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  CollectInfo(kernel_name_, inputs);

  return ret;
}

#define STRIDEDSLICE_GPU_REG(TYPEID_1, TYPEID_2, TYPE_1, TYPE_2) \
  KernelAttr()                                                   \
    .AddInputAttr(TYPEID_1)                                      \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddOutputAttr(TYPEID_1),                                    \
    &StridedSliceGpuKernelMod::LaunchKernel<TYPE_1, TYPE_2>

std::vector<std::pair<KernelAttr, StridedSliceGpuKernelMod::StridedSliceFunc>> StridedSliceGpuKernelMod::func_list_ = {
  {STRIDEDSLICE_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>, int64_t)},
  {STRIDEDSLICE_GPU_REG(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>, int64_t)},
};

std::vector<KernelAttr> StridedSliceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, StridedSliceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, StridedSlice, StridedSliceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
