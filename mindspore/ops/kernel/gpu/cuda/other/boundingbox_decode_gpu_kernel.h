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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_BOUNDINGBOX_DECODE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_BOUNDINGBOX_DECODE_GPU_KERNEL_H

#include <vector>
#include <utility>
#include <map>
#include <string>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/boundingbox_decode_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMinMaxShapeSize = 2;
class BoundingBoxDecodeGpuKernelMod : public NativeGpuKernelMod {
 public:
  BoundingBoxDecodeGpuKernelMod() : rois_size_(0), deltas_size_(0), bboxes_size_(0), wh_ratio_clip_(0.016) {}
  ~BoundingBoxDecodeGpuKernelMod() override = default;

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
  using BoundingBoxDecodeLaunchFunc = std::function<bool(
    BoundingBoxDecodeGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  BoundingBoxDecodeLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BoundingBoxDecodeLaunchFunc>> func_list_;
  size_t rois_size_;
  size_t deltas_size_;
  size_t bboxes_size_;
  std::vector<float> means_;
  std::vector<float> stds_;
  std::vector<int> max_shape_;
  float wh_ratio_clip_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_BOUNDINGBOX_DECODE_GPU_KERNEL_H
