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

#include "plugin/device/gpu/kernel/arrays/scalegrad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
size_t GetBaseTypeSize(TypeId type_id) {
  switch (type_id) {
    case kNumberTypeFloat16:
      return sizeof(half);
    case kNumberTypeFloat32:
      return sizeof(float);
    default:
      MS_LOG(EXCEPTION) << "For Scale Grad input type is error: " << type_id;
  }
}

size_t GetInputSize(const std::vector<int64_t> &input_shape, const TypeId &type_id) {
  size_t input_size = GetBaseTypeSize(type_id);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size *= LongToSize(input_shape[i]);
  }
  return input_size;
}
}  // namespace

template <typename T>
void ScaleGradGpuKernelMod::LaunchScaleGradPerGrad(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs, void *stream_ptr,
                                                   const half *scale_addr_half, const float *scale_addr_float,
                                                   size_t index) {
  T *input_addr = GetDeviceAddress<T>(inputs, index);
  T *output_addr = GetDeviceAddress<T>(outputs, index);
  if (scale_addr_half != nullptr) {
    ScaleGradKernel(outputs[index]->size / sizeof(T), input_addr, *scale_addr_half, output_addr,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    MS_EXCEPTION_IF_NULL(scale_addr_float);
    ScaleGradKernel(outputs[index]->size / sizeof(T), input_addr, *scale_addr_float, output_addr,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  }
}

bool ScaleGradGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  half *scale_addr_half = nullptr;
  float *scale_addr_float = nullptr;
  if (input_info_.back() == kNumberTypeFloat16) {
    scale_addr_half = GetDeviceAddress<half>(inputs, inputs.size() - 1);
  } else {
    scale_addr_float = GetDeviceAddress<float>(inputs, inputs.size() - 1);
  }

  for (size_t i = 0; i < inputs.size() - 1; i++) {
    switch (input_info_[i]) {
      case kNumberTypeFloat16: {
        LaunchScaleGradPerGrad<half>(inputs, outputs, stream_ptr, scale_addr_half, scale_addr_float, i);
        break;
      }
      case kNumberTypeFloat32: {
        LaunchScaleGradPerGrad<float>(inputs, outputs, stream_ptr, scale_addr_half, scale_addr_float, i);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the typeid cannot be " << input_info_[i];
    }
  }
  return true;
}

bool ScaleGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ScaleGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  kernel_name_ = kernel_ptr->name();
  auto input_size = inputs.size();
  for (size_t index = 0; index < input_size; index++) {
    auto type_id = inputs[index]->GetDtype();
    input_info_.push_back(type_id);
    auto size = GetInputSize(inputs[index]->GetShapeVector(), type_id);
    input_size_list_.push_back(size);
  }

  for (size_t index = 0; index < input_size - 1; index++) {
    output_size_list_.push_back(input_size_list_[index]);
  }

  return true;
}

MS_REG_GPU_KERNEL(ScaleGrad, ScaleGradGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
