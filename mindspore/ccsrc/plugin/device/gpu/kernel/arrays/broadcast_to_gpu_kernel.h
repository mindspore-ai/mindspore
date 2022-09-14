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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 8;
class BroadcastToGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastToGpuKernelMod() : kernel_name_("BroadcastTo") {}
  ~BroadcastToGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using BroadcastToLaunchFunc =
    std::function<bool(BroadcastToGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  std::string kernel_name_{};
  BroadcastToLaunchFunc kernel_func_;
  void ResetResource() noexcept;
  static std::vector<std::pair<KernelAttr, BroadcastToLaunchFunc>> func_list_;
  size_t input_size_;
  size_t output_size_;
  size_t input_type_size_;  // sizeof(T)
  std::vector<size_t> input_shape_ = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<size_t> output_shape_ = {1, 1, 1, 1, 1, 1, 1, 1};
  bool is_null_input_ = false;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BROADCAST_TO_GPU_KERNEL_H_
