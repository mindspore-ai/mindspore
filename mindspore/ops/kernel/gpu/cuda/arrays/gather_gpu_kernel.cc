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

#include "kernel/gpu/cuda/arrays/gather_gpu_kernel.h"
#include <memory>
#include "infer/ops_func_impl/gather.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool GatherGpuKernelMod::Init(const std::vector<kernel::KernelTensor *> &inputs,
                              const std::vector<kernel::KernelTensor *> &outputs) {
  if (auto ret = MatchKernelFunc(kernel_name_, inputs, outputs); !ret) {
    return ret;
  }
  input_type_size_ = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  indices_type_size_ = abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  return true;
}

int GatherGpuKernelMod::Resize(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shapes_ = inputs[kIndexZero]->GetShapeVector();
  indices_shapes_ = inputs[kIndexOne]->GetShapeVector();
  output_shapes_ = outputs[kIndexZero]->GetShapeVector();

  axis_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  if (inputs.size() == kSizeFour) {
    batch_dims_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
    if (batch_dims_ < 0) {
      batch_dims_ += SizeToLong(indices_shapes_.size());
    }
  }
  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shapes_, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  Reshape();
  return KRET_OK;
}

template <typename T, typename S>
bool GatherGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto input_dim1 = input_shapes_[IntToSize(axis_)];
  auto status = Gather(input_addr, indices_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
                       dims_[kIndex3], LongToSize(input_dim1), reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define REG_INDEX(INPUT_DT, INDEX_DT, INPUT_T, INDEX_T)   \
  {                                                       \
    KernelAttr()                                          \
      .AddInputAttr(INPUT_DT)                             \
      .AddInputAttr(INDEX_DT)                             \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)  \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)  \
      .AddOutputAttr(INPUT_DT),                           \
      &GatherGpuKernelMod::LaunchKernel<INPUT_T, INDEX_T> \
  }

#define GATHER_GPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

template <typename T>
using Complex = mindspore::utils::Complex<T>;

const std::vector<std::pair<KernelAttr, GatherGpuKernelMod::KernelRunFunc>> &GatherGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GatherGpuKernelMod::KernelRunFunc>> func_list = {
    GATHER_GPU_REGISTER(kNumberTypeComplex64, Complex<float>),
    GATHER_GPU_REGISTER(kNumberTypeComplex128, Complex<double>),
    GATHER_GPU_REGISTER(kNumberTypeFloat16, half),
    GATHER_GPU_REGISTER(kNumberTypeFloat32, float),
    GATHER_GPU_REGISTER(kNumberTypeFloat64, double),
    GATHER_GPU_REGISTER(kNumberTypeInt8, uchar),
    GATHER_GPU_REGISTER(kNumberTypeInt16, int16_t),
    GATHER_GPU_REGISTER(kNumberTypeInt32, int32_t),
    GATHER_GPU_REGISTER(kNumberTypeInt64, int64_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt8, uint8_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt16, uint16_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt32, uint32_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt64, uint64_t),
    GATHER_GPU_REGISTER(kNumberTypeBool, bool)};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Gather, GatherGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
