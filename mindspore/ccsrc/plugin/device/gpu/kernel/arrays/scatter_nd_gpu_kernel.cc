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
const std::vector<std::pair<KernelAttr, ScatterNdGpuKernelMod::KernelRunFunc>> &ScatterNdGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ScatterNdGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ScatterNdGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ScatterNdGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ScatterNdGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ScatterNdGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ScatterNdGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ScatterNdGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ScatterNdGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ScatterNdGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &ScatterNdGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &ScatterNdGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ScatterNdGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ScatterNdGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  };
  return func_list;
}

void ScatterNdGpuKernelMod::FreeResource() {
  if (indices_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(indices_stride_);
    indices_stride_ = nullptr;
  }
  if (work_shape_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(work_shape_);
    work_shape_ = nullptr;
  }
}

template <typename T, typename S>
bool ScatterNdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  S *indices = GetDeviceAddress<S>(inputs, 0);
  T *update = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);

  if (!memcpy_flag_) {
    const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
    const size_t vec_work_len = sizeof(S) * attr_shape_.size();
    std::vector<S> tmp_ind_stride;
    (void)std::transform(vec_indices_stride_.begin(), vec_indices_stride_.end(), std::back_inserter(tmp_ind_stride),
                         [](size_t x) { return static_cast<S>(x); });
    std::vector<S> tmp_work_shape;
    (void)std::transform(attr_shape_.begin(), attr_shape_.end(), std::back_inserter(tmp_work_shape),
                         [](int64_t x) { return static_cast<S>(x); });
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(static_cast<S *>(indices_stride_), &tmp_ind_stride[0], indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpy for indices_stride failed in ScatterNdGpuKernelMod::LaunchKernel.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(static_cast<S *>(work_shape_), &tmp_work_shape[0], vec_work_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpy for work_shape failed in ScatterNdGpuKernelMod::LaunchKernel.");
    memcpy_flag_ = true;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(output, static_cast<T>(0.0), output_size_list_[0], reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemSet failed in ScatterNdGpuKernelMod::LaunchKernel.");

  const size_t input_size = input_size_list_[kIndex1] / sizeof(T);
  const size_t output_size = output_size_list_[kIndex0] / sizeof(T);

  ScatterNd(indices, update, output, block_size_, input_size, output_size, indices_dim_0_, indices_dim_1_,
            static_cast<S *>(indices_stride_), static_cast<S *>(work_shape_),
            reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

bool ScatterNdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  auto attr = base_operator->GetPrim()->GetAttr(kAttrShape);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr \"shape\" is not found in kernel 'ScatterNd'.";
    return false;
  }
  attr_shape_ = GetValue<std::vector<int64_t>>(attr);
  return true;
}

int ScatterNdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  memcpy_flag_ = false;

  CalSize(inputs, outputs);
  auto indices_unit_size = abstract::TypeIdSize(inputs[0]->GetDtype());
  const size_t indices_len = indices_unit_size * vec_indices_stride_.size();
  indices_stride_ = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
  if (indices_stride_ == nullptr) {
    MS_LOG(EXCEPTION) << "For 'ScatterNd', the memory alloc of indices_stride_work must be successful, but failed."
                      << " got size: " << indices_len;
  }

  const size_t vec_work_len = indices_unit_size * attr_shape_.size();
  work_shape_ = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(vec_work_len);
  if (work_shape_ == nullptr) {
    MS_LOG(EXCEPTION) << "For 'ScatterNd', the memory alloc of indices_stride_work must be successful, but failed."
                      << " got size: " << vec_work_len;
  }

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
