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

#include "mindspore/core/ops/base_operator.h"
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/sparse/sparse_to_dense_v2_gpu_kernel.h"

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

bool SparseToDenseV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::SparseToDenseV2>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  validate_indices_ = kernel_ptr_->get_validate_indices();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  indice_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  value_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  return true;
}

int SparseToDenseV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  indices_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_shape_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                      inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<size_t> input_shape_values = std::vector<size_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                               inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  indices_dims_ = indices_shape_.size();
  ndims = indices_shape_.size() > 1 ? indices_shape_[1] : 1;
  num_elems = indices_shape_.size() > 0 ? indices_shape_[0] : 1;
  values_size_ = input_shape_values[0];
  output_elements = 1;
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    output_elements *= output_shape[i];
  }
  input_elements_indices = std::accumulate(indices_shape_.begin(), indices_shape_.end(), 1, std::multiplies<size_t>());
  input_elements_values =
    std::accumulate(input_shape_values.begin(), input_shape_values.end(), 1, std::multiplies<size_t>());
  input_elements_output_shape =
    std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  size_t input_size_indices = input_elements_indices * indice_size_;
  size_t input_size_values = input_elements_values * value_size_;
  size_t input_size_output_shape = input_elements_output_shape * indice_size_;
  size_t output_size = output_elements * value_size_;
  input_size_list_.push_back(input_size_indices);
  input_size_list_.push_back(input_size_values);
  input_size_list_.push_back(input_size_output_shape);
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

void SparseToDenseV2GpuKernelMod::ResetResource() noexcept {
  output_elements = 1;
  input_elements_indices = 0;
  input_elements_values = 0;
  input_elements_output_shape = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename I, typename T>
void SparseToDenseV2GpuKernelMod::CheckValidateTwoDim(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &workspace,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  I *input_indices = GetDeviceAddress<I>(inputs, kIndex0);
  I *indices_addr = reinterpret_cast<I *>(malloc(input_elements_indices * indice_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_addr, input_indices, input_elements_indices * indice_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync indices failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }

  I *input_output_shape = GetDeviceAddress<I>(inputs, kIndex1);
  I *output_shape_addr = reinterpret_cast<I *>(malloc(input_elements_output_shape * indice_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_shape_addr, input_output_shape, input_elements_output_shape * indice_size_,
                    cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream_)),
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
    if (indices_addr[index] < 0 || indices_addr[index] >= output_shape_addr[index]) {
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
      if (indices_addr[index1] < 0 || indices_addr[index1] >= output_shape_addr[j]) {
        valid = false;
      }
      I diff = indices_addr[index1] - indices_addr[index2];
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
void SparseToDenseV2GpuKernelMod::CheckValidateOneDim(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &workspace,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseV2OutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  I *input_indices = GetDeviceAddress<I>(inputs, kIndex0);
  I *indices_addr = reinterpret_cast<I *>(malloc(input_elements_indices * indice_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_addr, input_indices, input_elements_indices * indice_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync indices failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }

  I *input_output_shape = GetDeviceAddress<I>(inputs, kIndex1);
  I *output_shape_addr = reinterpret_cast<I *>(malloc(input_elements_output_shape * indice_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_shape_addr, input_output_shape, input_elements_output_shape * indice_size_,
                    cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync dense_shape failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }
  bool valid = true;
  bool different = false;
  bool increasing = true;
  if (indices_addr[0] < 0 || indices_addr[0] > output_shape_addr[0]) {
    valid = false;
  }
  for (size_t i = 1; i < indices_shape_[0]; ++i) {
    if (indices_addr[i] < 0 || indices_addr[i] >= output_shape_addr[0]) {
      valid = false;
    }
    I diff = indices_addr[i] - indices_addr[i - 1];
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
bool SparseToDenseV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
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

  T *default_value_data = reinterpret_cast<T *>(malloc(value_size_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(default_value_data, input_default_value, value_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseToDenseV2', cudaMemcpyAsync default_value failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseToDenseV2', cuda Stream Sync Failed.");
  }

  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  auto status = CallSetDefaultValue(default_value_data[0], output_elements, output, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CallSparseToDense(input_indices, input_values, num_elems, input_elements_values, input_output_shape, ndims,
                             output, device_id_, cuda_stream);
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
