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

#include "plugin/device/gpu/kernel/arrays/scatter_nd_gpu_kernel.h"
#include <algorithm>

namespace mindspore {
namespace kernel {
#define DTYPE_REGISTER(INDICES, UPDATES, SHAPE, OUTPUT, T, S)                                           \
  {                                                                                                     \
    KernelAttr().AddInputAttr(INDICES).AddInputAttr(UPDATES).AddInputAttr(SHAPE).AddOutputAttr(OUTPUT), \
      &ScatterNdGpuKernelMod::LaunchKernel<T, S>                                                        \
  }

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
  };
  return func_list;
}

template <typename T, typename S>
bool ScatterNdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  S *indices = GetDeviceAddress<S>(inputs, 0);
  T *update = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  S *indices_stride = GetDeviceAddress<S>(workspace, kIndex0);
  S *work_shape = GetDeviceAddress<S>(workspace, kIndex1);

  const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
  const size_t vec_work_len = sizeof(S) * attr_shape_.size();
  std::vector<S> tmp_ind_stride;
  (void)std::transform(vec_indices_stride_.begin(), vec_indices_stride_.end(), std::back_inserter(tmp_ind_stride),
                       [](size_t x) { return static_cast<S>(x); });
  std::vector<S> tmp_work_shape;
  (void)std::transform(attr_shape_.begin(), attr_shape_.end(), std::back_inserter(tmp_work_shape),
                       [](int64_t x) { return static_cast<S>(x); });

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_stride, &tmp_ind_stride[0], indices_len, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpy for indices_stride failed in "
    "ScatterNdGpuKernelMod::LaunchKernel.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(work_shape, &tmp_work_shape[0], vec_work_len, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpy for work_shape failed in "
    "ScatterNdGpuKernelMod::LaunchKernel.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(output, static_cast<T>(0.0), output_size_list_[0], reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemSet failed in ScatterNdGpuKernelMod::LaunchKernel.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");

  const size_t input_size = input_size_list_[kIndex1] / sizeof(T);
  const size_t output_size = output_size_list_[kIndex0] / sizeof(T);
  ScatterNd(indices, update, output, block_size_, input_size, output_size, indices_dim_0_, indices_dim_1_,
            indices_stride, work_shape, reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

bool ScatterNdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  return true;
}

int ScatterNdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  if (!TryGetIntValue(inputs, kShapeIndex_, kernel_name_, &attr_shape_)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << "can't get shape input!";
    return KRET_RESIZE_FAILED;
  }

  CalSize(inputs, outputs);
  auto indices_unit_size = abstract::TypeIdSize(inputs[0]->GetDtype());
  const size_t indices_len = indices_unit_size * vec_indices_stride_.size();
  workspace_size_list_.push_back(indices_len);
  const size_t vec_work_len = indices_unit_size * attr_shape_.size();
  workspace_size_list_.push_back(vec_work_len);

  return KRET_OK;
}

void ScatterNdGpuKernelMod::CalSize(const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
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
