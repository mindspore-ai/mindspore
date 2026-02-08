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

#include "kernel/gpu/cuda/arrays/scalegrad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ScaleGradGpuKernelMod::LaunchScaleGradPerGrad(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr,
                                                   const half *scale_addr_half, const float *scale_addr_float,
                                                   size_t index) {
  T *input_addr = GetDeviceAddress<T>(inputs, index);
  T *output_addr = GetDeviceAddress<T>(outputs, index);
  cudaError_t status = cudaErrorNotReady;
  if (scale_addr_half != nullptr) {
    status = ScaleGradKernel(outputs[index]->size() / sizeof(T), input_addr, *scale_addr_half, output_addr,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    MS_EXCEPTION_IF_NULL(scale_addr_float);
    status = ScaleGradKernel(outputs[index]->size() / sizeof(T), input_addr, *scale_addr_float, output_addr,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
}

bool ScaleGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
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

bool ScaleGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto input_size = inputs.size();
  for (size_t index = 0; index < input_size; index++) {
    auto type_id = inputs[index]->dtype_id();
    input_info_.push_back(type_id);
    if (index < input_size - 1) {
      output_size_list_.push_back(inputs[index]->size());
    }
  }

  return true;
}

MS_REG_GPU_KERNEL(ScaleGrad, ScaleGradGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
