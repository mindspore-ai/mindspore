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

#include "plugin/device/gpu/kernel/arrays/tensor_scatter_elements_gpu_kernel.h"
#include "mindspore/core/abstract/utils.h"

namespace mindspore {
namespace kernel {
void TensorScatterElementsGpuKernelMod::FreeResource() {
  if (d_indices_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_indices_stride_);
    d_indices_stride_ = nullptr;
  }

  if (d_output_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_output_stride_);
    d_output_stride_ = nullptr;
  }
}

void TensorScatterElementsGpuKernelMod::MallocResource() {
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

void TensorScatterElementsGpuKernelMod::GetSize() {
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

bool TensorScatterElementsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::TensorScatterElements>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  std::string reduction = kernel_ptr->get_reduction();
  if (reduction == "none") {
    type_ = TensorScatterElementsReductionType::REDUCTION_ASSIGNMENT;
  } else if (reduction == "add") {
    type_ = TensorScatterElementsReductionType::REDUCTION_ADD;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduction type: " << reduction << " not support now.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  indices_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  return true;
}

int TensorScatterElementsGpuKernelMod::ShapeCheck() {
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got " << input_dims_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indices_shape_ != updates_shape_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'indice' and the shape of 'update' should be same, but got "
                  << "indice shape: " << indices_shape_ << "; "
                  << "update shape: " << updates_shape_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_dims_ != indices_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x', 'indice' and 'update' should be same, but got "
                  << "input_x dims: " << input_dims_ << "; "
                  << "indice dims: " << indices_shape_.size() << "; "
                  << "update dims: " << updates_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int TensorScatterElementsGpuKernelMod::AxisCheck() {
  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the 'axis' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape_[i] < indices_shape_[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the indices dims should be less than input dims, but got indice dim is: "
                    << indices_shape_[i] << " at axis: " << i << ", while input dim is:" << input_shape_[i];
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

int TensorScatterElementsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }
  FreeResource();
  sync_resource_ = true;

  input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                     inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  indices_shape_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  updates_shape_ = std::vector<size_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());

  input_dims_ = input_shape_.size();

  ret = ShapeCheck();
  if (ret != KRET_OK) {
    return ret;
  }

  if (base_operator->HasAttr(kAttrAxis)) {
    axis_ = GetValue<int64_t>(base_operator->GetAttr(kAttrAxis));
    if (axis_ < 0) {
      axis_ += input_dims_;
    }
  }

  ret = AxisCheck();
  if (ret != KRET_OK) {
    return ret;
  }

  GetSize();
  MallocResource();
  return KRET_OK;
}

template <typename T, typename S>
bool TensorScatterElementsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *updates = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  if (sync_resource_) {
    const size_t indices_stride_len = sizeof(size_t) * indices_stride_.size();
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_indices_stride_, indices_stride_.data(), indices_stride_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in TensorScatterElementsGpuKernelMod::Launch.");

    const size_t output_stride_len = sizeof(size_t) * output_stride_.size();
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_output_stride_, output_stride_.data(), output_stride_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in TensorScatterElementsGpuKernelMod::Launch.");

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

#define SCATTER_ELEMENTS_GPU_REG(MS_T, MS_S, T, S)                                           \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &TensorScatterElementsGpuKernelMod::LaunchKernel<T, S>

std::vector<std::pair<KernelAttr, TensorScatterElementsGpuKernelMod::TensorScatterElementsFunc>>
  TensorScatterElementsGpuKernelMod::func_list_ = {
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
    {SCATTER_ELEMENTS_GPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
};

std::vector<KernelAttr> TensorScatterElementsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TensorScatterElementsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterElements, TensorScatterElementsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
