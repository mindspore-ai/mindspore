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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SCALEGRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SCALEGRAD_GPU_KERNEL_H

#include <vector>
#include <string>
#include <memory>
#include "mindspore/ops/infer/cxx_api/scale_grad_fusion.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/scale_grad_impl.cuh"

namespace mindspore {
namespace kernel {
class ScaleGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ScaleGradGpuKernelMod() { kernel_name_ = "ScaleGrad"; }
  ~ScaleGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  template <typename T>
  void LaunchScaleGradPerGrad(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                              void *stream_ptr, const half *scale_addr_half, const float *scale_addr_float,
                              size_t index);
  std::vector<TypeId> input_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPLIT_GPU_KERNEL_H
