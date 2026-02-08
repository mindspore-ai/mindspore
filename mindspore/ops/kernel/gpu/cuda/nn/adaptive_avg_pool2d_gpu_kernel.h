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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/adaptive_avg_pool2d_impl.cuh"

namespace mindspore {
namespace kernel {

class AdaptiveAvgPool2DKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<AdaptiveAvgPool2DKernelMod> {
 public:
  AdaptiveAvgPool2DKernelMod() = default;
  ~AdaptiveAvgPool2DKernelMod() override = default;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  uint len_{0};
  uint input_height_{0};
  uint input_width_{0};
  uint output_height_{0};
  uint output_width_{0};
  uint size_{0};
  bool is_null_input_{false};
  std::string kernel_name_{"AdaptiveAvgPool2D"};
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  //  MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_
