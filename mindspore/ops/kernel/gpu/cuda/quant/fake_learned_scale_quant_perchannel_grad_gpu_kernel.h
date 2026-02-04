/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PER_CHANNEL_GRAD_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PER_CHANNEL_GRAD_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class FakeLearnedScaleQuantPerChannelGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  FakeLearnedScaleQuantPerChannelGradGpuKernelMod();
  ~FakeLearnedScaleQuantPerChannelGradGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  size_t input_size_;
  size_t workspace_size_;

  int quant_num_;
  int quant_delay_;
  int global_step_;
  bool neg_trunc_;
  int num_channels_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PER_CHANNEL_GRAD_GPUKERNEL_H_
