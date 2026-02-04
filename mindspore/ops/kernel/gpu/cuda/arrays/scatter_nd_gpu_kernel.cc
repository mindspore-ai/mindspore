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

#include "kernel/gpu/cuda/arrays/scatter_nd_gpu_kernel.h"
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
#define DTYPE_REGISTER(INDICES, UPDATES, SHAPE, OUTPUT, T, S) \
  {                                                           \
    KernelAttr()                                              \
      .AddInputAttr(INDICES)                                  \
      .AddInputAttr(UPDATES)                                  \
      .AddInputAttr(kObjectTypeTuple, SHAPE)                  \
      .AddOutputAttr(OUTPUT),                                 \
      &ScatterNdGpuKernelMod::LaunchKernel<T, S>              \
  }

template <typename T>
using Complex = mindspore::utils::Complex<T>;

const std::vector<std::pair<KernelAttr, ScatterNdGpuKernelMod::KernelRunFunc>> &ScatterNdGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ScatterNdGpuKernelMod::KernelRunFunc>> func_list = {
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, half, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, half, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, half, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, bool, int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, bool, int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, bool, int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, Complex<float>,
                   int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, Complex<float>,
                   int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, Complex<float>,
                   int64_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, Complex<double>,
                   int16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, Complex<double>,
                   int32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, Complex<double>,
                   int64_t),
  };
  return func_list;
}

template <typename T, typename S>
bool ScatterNdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  S *indices = GetDeviceAddress<S>(inputs, 0);
  T *update = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  ScatterNdInfo<S> info;
  for (size_t i = 0; i < vec_indices_stride_.size(); ++i) {
    info.indices_stride[i] = static_cast<S>(vec_indices_stride_[i]);
  }
  for (size_t i = 0; i < attr_shape_.size(); ++i) {
    info.shape[i] = static_cast<S>(attr_shape_[i]);
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(output, 0, output_size_list_[0], reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemSet failed in ScatterNdGpuKernelMod::LaunchKernel.");

  const size_t input_size = inputs[kIndex1]->size() / sizeof(T);
  const size_t output_size = output_size_list_[kIndex0] / sizeof(T);
  auto status = ScatterNd(indices, update, output, block_size_, input_size, output_size, indices_dim_0_, indices_dim_1_,
                          info, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool ScatterNdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ScatterNdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  attr_shape_ = inputs[kShapeIndex_]->GetValueWithCheck<ShapeVector>();
  CalSize(inputs, outputs);
  return KRET_OK;
}

void ScatterNdGpuKernelMod::CalSize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto indices_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();

  // calculate indices dim 0/1
  indices_dim_0_ = indices_shape[0];
  indices_dim_1_ = indices_shape.back();

  // calculate block_size
  block_size_ = 1;
  for (size_t i = indices_dim_1_; i < output_shape.size(); i++) {
    block_size_ *= LongToSize(output_shape[i]);
  }

  // calculate indices_stride
  vec_indices_stride_.clear();
  vec_indices_stride_.resize(indices_dim_1_, 0);
  vec_indices_stride_[indices_dim_1_ - 1] = block_size_;

  for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
    vec_indices_stride_[i - 1] = vec_indices_stride_[i] * LongToSize(output_shape[i]);
  }
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNd, ScatterNdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
