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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_encode_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class BoundingBoxEncodeGpuKernelMod : public NativeGpuKernelMod {
 public:
  BoundingBoxEncodeGpuKernelMod() : anchor_size_(0), groundtruth_size_(0), deltas_size_(0) {}
  ~BoundingBoxEncodeGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using BoundingBoxEncodeLaunchFunc =
    std::function<bool(BoundingBoxEncodeGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  BoundingBoxEncodeLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BoundingBoxEncodeLaunchFunc>> func_list_;
  size_t anchor_size_;
  size_t groundtruth_size_;
  size_t deltas_size_;
  std::vector<float> means_;
  std::vector<float> stds_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H
