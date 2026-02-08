/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MEMCPY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MEMCPY_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <map>
#include <vector>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MemcpyGpuKernelMod : public NativeGpuKernelMod {
 public:
  MemcpyGpuKernelMod() = default;
  explicit MemcpyGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~MemcpyGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override { return true; }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

  // The input addresses that are not used in the kernel launch.
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

 private:
  size_t input_size_{0};
  bool is_null_input_{false};
  std::string kernel_type_{"Unknown"};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MEMCPY_GPU_KERNEL_H_
