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

#include "kernel/gpu/cuda/arrays/scatter_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ScatterGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ == kTensorScatterElements) {
    input_name_ = "data";
    axis_name_ = "axis";
    index_name_ = "indices";
    src_name_ = "updates";
    axis_idx_ = kIndex3;
    index_idx_ = kIndex1;
    src_idx_ = kIndex2;
  } else if (kernel_name_ != kScatter) {
    MS_LOG(ERROR) << "Need to be " << kScatter << " or " << kTensorScatterElements << ", but got kernel name as "
                  << kernel_name_;
    return false;
  }
  type_ = inputs[reduce_idx_]->GetValueWithCheck<Reduce>();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  indices_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(index_idx_).dtype);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(input_idx_).dtype);

  return true;
}

void ScatterGpuKernelMod::MallocResource() {
  const size_t indices_stride_len = sizeof(size_t) * indices_stride_.size();
  d_indices_stride_ =
    static_cast<size_t *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_stride_len));
  if (d_indices_stride_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory alloc of d_indices_stride_ must be successful, but failed, got size: "
                      << indices_stride_len;
  }

  const size_t output_stride_len = sizeof(size_t) * output_stride_.size();
  d_output_stride_ =
    static_cast<size_t *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(output_stride_len));
  if (d_output_stride_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory alloc of d_output_stride_ must be successful, but failed, got size: "
                      << output_stride_len;
  }
}

void ScatterGpuKernelMod::FreeResource() {
  if (d_indices_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_indices_stride_);
    d_indices_stride_ = nullptr;
  }

  if (d_output_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_output_stride_);
    d_output_stride_ = nullptr;
  }
}

void ScatterGpuKernelMod::GetSize() {
  input_byte_size_ = data_unit_size_;
  for (const auto &shape_item : input_shape_) {
    input_byte_size_ *= shape_item;
  }
  indices_byte_size_ = indices_unit_size_;
  for (const auto &shape_item : indices_shape_) {
    indices_byte_size_ *= shape_item;
  }
  // calculate indices_stride
  indices_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    indices_stride_[i - 1] = indices_stride_[i] * indices_shape_[i];
  }

  // calculate output_stride
  output_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    output_stride_[i - 1] = output_stride_[i] * output_shape_[i];
  }

  input_axis_size_ = input_shape_[axis_];
}

int ScatterGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }
  FreeResource();
  sync_resource_ = true;

  input_shape_ = std::vector<size_t>(inputs.at(input_idx_)->GetDeviceShapeVector().begin(),
                                     inputs.at(input_idx_)->GetDeviceShapeVector().end());
  indices_shape_ = std::vector<size_t>(inputs.at(index_idx_)->GetDeviceShapeVector().begin(),
                                       inputs.at(index_idx_)->GetDeviceShapeVector().end());
  src_shape_ = std::vector<size_t>(inputs.at(src_idx_)->GetDeviceShapeVector().begin(),
                                   inputs.at(src_idx_)->GetDeviceShapeVector().end());
  output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                      outputs.at(kIndex0)->GetDeviceShapeVector().end());

  input_dims_ = input_shape_.size();

  ret = ShapeCheck();
  if (ret != KRET_OK) {
    return ret;
  }

  axis_ = inputs[axis_idx_]->GetValueWithCheck<int64_t>();
  if (axis_ < 0) {
    axis_ += static_cast<int64_t>(input_dims_);
  }

  ret = AxisCheck();
  if (ret != KRET_OK) {
    return ret;
  }

  GetSize();
  MallocResource();
  return KRET_OK;
}

int ScatterGpuKernelMod::ShapeCheck() {
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of '" << input_name_
                  << "' should be greater than or equal to 1, but got " << input_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indices_shape_ != src_shape_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of '" << index_name_ << "' and the shape of '"
                  << src_name_ << "' should be same, but got " << index_name_ << " shape: " << indices_shape_ << "; "
                  << src_name_ << " shape: " << src_shape_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_dims_ != indices_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of '" << input_name_ << "' and '" << index_name_
                  << "' should be same, but got " << input_name_ << " dims: " << input_dims_ << "; " << index_name_
                  << " dims: " << indices_shape_.size() << ". ";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int ScatterGpuKernelMod::AxisCheck() {
  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the '" << axis_name_
                  << "' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape_[i] < indices_shape_[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << index_name_
                    << " dims should be less than input dims, but got " << index_name_
                    << " dim is: " << indices_shape_[i] << " at axis: " << i
                    << ", while input dim is:" << input_shape_[i];
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

template <typename T, typename S>
bool ScatterGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  T *input = GetDeviceAddress<T>(inputs, input_idx_);
  S *indices = GetDeviceAddress<S>(inputs, index_idx_);
  T *updates = GetDeviceAddress<T>(inputs, src_idx_);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  if (sync_resource_) {
    const size_t indices_stride_len = sizeof(size_t) * indices_stride_.size();
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_indices_stride_, indices_stride_.data(), indices_stride_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in ScatterGpuKernelMod::Launch.");

    const size_t output_stride_len = sizeof(size_t) * output_stride_.size();
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_output_stride_, output_stride_.data(), output_stride_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in ScatterGpuKernelMod::Launch.");

    sync_resource_ = false;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output, input, input_byte_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "cudaMemcpy output failed");

  auto status = TensorScatterElements(type_, input_dims_, indices_byte_size_ / indices_unit_size_, indices, updates,
                                      output, axis_, input_axis_size_, d_indices_stride_, d_output_stride_, device_id_,
                                      reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define SCATTER_GPU_REG(MS_T, MS_S, T, S)              \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddInputAttr(MS_S)                                \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
    &ScatterGpuKernelMod::LaunchKernel<T, S>

#define SCATTER_ELEMENTS_GPU_REG(MS_T, MS_S, T, S)     \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(MS_S)                                \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
    &ScatterGpuKernelMod::LaunchKernel<T, S>

void ScatterGpuKernelMod::GetFuncList() {
  if (kernel_name_ == kScatter) {
    func_list_ = {{SCATTER_GPU_REG(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
                  {SCATTER_GPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
                  {SCATTER_GPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
                  {SCATTER_GPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int, int)},
                  {SCATTER_GPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int)},
                  {SCATTER_GPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int)},
                  {SCATTER_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
                  {SCATTER_GPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)}};
  } else {
    func_list_ = {{SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
                  {SCATTER_ELEMENTS_GPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)}};
  }
}

std::vector<KernelAttr> ScatterGpuKernelMod::GetOpSupport() {
  GetFuncList();
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScatterFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Scatter, ScatterGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterElements, ScatterGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
