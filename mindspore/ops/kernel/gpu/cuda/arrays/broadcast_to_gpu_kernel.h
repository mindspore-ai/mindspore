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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BROADCAST_TO_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BROADCAST_TO_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <utility>
#include <algorithm>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/broadcast_to_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 8;
class BroadcastToGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastToGpuKernelMod() : kernel_name_("BroadcastTo") {}
  ~BroadcastToGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using BroadcastToLaunchFunc = std::function<bool(
    BroadcastToGpuKernelMod *, const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  std::string kernel_name_{};
  BroadcastToLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BroadcastToLaunchFunc>> func_list_;
  bool is_broadcast_{false};
  std::vector<int64_t> simplified_inp_shape_;
  std::vector<int64_t> simplified_out_shape_;
  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BROADCAST_TO_GPU_KERNEL_H_
