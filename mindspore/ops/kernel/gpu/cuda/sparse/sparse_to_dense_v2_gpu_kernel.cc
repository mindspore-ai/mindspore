/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/sparse/sparse_to_dense_v2_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseToDenseV2InputsNum = 4;
constexpr size_t kSparseToDenseV2OutputsNum = 1;
constexpr size_t kSparseToDenseV2First = 0;
constexpr size_t kSparseToDenseV2Second = 1;
constexpr size_t kSparseToDenseV2Third = 2;
constexpr size_t kSparseToDenseV2Fourth = 3;
constexpr size_t kSparseToDenseV2TwoDims = 2;
constexpr size_t kSparseToDenseV2OneDim = 1;
constexpr size_t kSparseToDenseV2ZeroDim = 0;
}  // namespace

bool SparseToDenseV2GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  validate_indices_ = GetValue<bool>(primitive_->GetAttr("validate_indices"));
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  value_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  return true;
}

int SparseToDenseV2GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  indices_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeVector().end());
  output_shape_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                      inputs.at(kIndex1)->GetDeviceShapeVector().end());
  std::vector<size_t> input_shape_values = std::vector<size_t>(inputs.at(kIndex2)->GetDeviceShapeVector().begin(),
                                                               inputs.at(kIndex2)->GetDeviceShapeVector().end());
  indices_dims_ = indices_shape_.size();
  ndims = indices_shape_.size() > 1 ? indices_shape_[1] : 1;
  num_elems = indices_shape_.size() > 0 ? indices_shape_[0] : 1;
  values_size_ = input_shape_values[0];
  output_elements = 1;
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    output_elements *= output_shape[i];
  }
  indices_num_ = std::accumulate(indices_shape_.begin(), indices_shape_.end(), 1, std::multiplies<size_t>());
  values_num_ = std::accumulate(input_shape_values.begin(), input_shape_values.end(), 1, std::multiplies<size_t>());
  output_num_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  size_t output_size = output_elements * value_size_;
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

void SparseToDenseV2GpuKernelMod::ResetResource() noexcept {
  output_elements = 1;
  indices_num_ = 0;
  values_num_ = 0;
  output_num_ = 0;
  is_null_input_ = false;
  output_size_list_.clear();
}

template <typename I, typename T>
void SparseToDenseV2GpuKernelMod::CheckValidateTwoDim(const std::vector<kernel::KernelTensor *> &inputs,
                                                      const std::vector<kernel::KernelTensor *> &workspace,
                                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  if (outputs[0]->size() == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  I *input_indices = GetDeviceAddress<I>(inputs, kIndex0);
  std::vector<I> indices_host;
  indices_host.resize(indices_num_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_host.data(), input_indices, indices_num_ * sizeof(I), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync indices failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }

  I *input_output_shape = GetDeviceAddress<I>(inputs, kIndex1);
  std::vector<I> output_shape_host;
  output_shape_host.resize(output_num_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_shape_host.data(), input_output_shape, output_num_ * sizeof(I), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync dense_shape failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }
  bool valid = true;
  bool different = false;
  bool increasing = true;
  for (size_t k = 0; k < indices_shape_[1]; ++k) {
    size_t index = k;
    if (indices_host[index] < 0 || indices_host[index] >= output_shape_host[index]) {
      valid = false;
    }
  }
  if (!valid) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is out of bounds.";
  }
  for (size_t i = 1; i < indices_shape_[0]; ++i) {
    for (size_t j = 0; j < indices_shape_[1]; ++j) {
      size_t index1 = i * indices_shape_[1] + j;
      size_t index2 = (i - 1) * indices_shape_[1] + j;
      if (indices_host[index1] < 0 || indices_host[index1] >= output_shape_host[j]) {
        valid = false;
      }
      I diff = indices_host[index1] - indices_host[index2];
      if (diff > 0) {
        different = true;
      }
      if (!different && diff < 0) {
        increasing = false;
      }
    }
    if (!valid) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is out of bounds.";
    }
    if (!increasing) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is out of order.";
    }
    if (!different) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is repeated";
    }
  }
}

template <typename I, typename T>
void SparseToDenseV2GpuKernelMod::CheckValidateOneDim(const std::vector<kernel::KernelTensor *> &inputs,
                                                      const std::vector<kernel::KernelTensor *> &workspace,
                                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  if (outputs[0]->size() == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  I *input_indices = GetDeviceAddress<I>(inputs, kIndex0);
  std::vector<I> indices_host;
  indices_host.resize(indices_num_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_host.data(), input_indices, indices_num_ * sizeof(I), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync indices failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }

  I *input_output_shape = GetDeviceAddress<I>(inputs, kIndex1);
  std::vector<I> output_shape_host;
  output_shape_host.resize(output_num_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_shape_host.data(), input_output_shape, output_num_ * sizeof(I), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync dense_shape failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }
  bool valid = true;
  bool different = false;
  bool increasing = true;
  if (indices_host[0] < 0 || indices_host[0] > output_shape_host[0]) {
    valid = false;
  }
  for (size_t i = 1; i < indices_shape_[0]; ++i) {
    if (indices_host[i] < 0 || indices_host[i] >= output_shape_host[0]) {
      valid = false;
    }
    I diff = indices_host[i] - indices_host[i - 1];
    if (diff > 0) {
      different = true;
    }
    if (!different && diff < 0) {
      increasing = false;
    }
    if (!valid) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is out of bounds.";
    }
    if (!increasing) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is out of order.";
    }
    if (!different) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices is repeated";
    }
  }
}

template <typename I, typename T>
bool SparseToDenseV2GpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (validate_indices_ == true && indices_dims_ == kSparseToDenseV2TwoDims) {
    (void)SparseToDenseV2GpuKernelMod::CheckValidateTwoDim<I, T>(inputs, workspace, outputs);
  } else if (validate_indices_ == true && indices_dims_ == kSparseToDenseV2OneDim) {
    (void)SparseToDenseV2GpuKernelMod::CheckValidateOneDim<I, T>(inputs, workspace, outputs);
  }
  I *input_indices = GetDeviceAddress<I>(inputs, kIndex0);
  I *input_output_shape = GetDeviceAddress<I>(inputs, kIndex1);
  T *input_values = GetDeviceAddress<T>(inputs, kIndex2);
  T *input_default_value = GetDeviceAddress<T>(inputs, kIndex3);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  auto status = CallSetDefaultValue(input_default_value, output_elements, output, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CallSparseToDense(input_indices, input_values, num_elems, values_num_, input_output_shape, ndims, output,
                             device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseToDenseV2GpuKernelMod::SparseToDenseV2LaunchFunc>>
  SparseToDenseV2GpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeBool)
                                                .AddInputAttr(kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeBool),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, bool>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, int8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, int16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, uint8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, uint16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int32_t, double>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeBool)
                                                .AddInputAttr(kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeBool),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, bool>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, int8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, int16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, uint8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, uint16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &SparseToDenseV2GpuKernelMod::LaunchKernel<int64_t, double>}};

std::vector<KernelAttr> SparseToDenseV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseToDenseV2GpuKernelMod::SparseToDenseV2LaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseToDenseV2, SparseToDenseV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
