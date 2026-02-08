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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_IOU_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_IOU_GPU_KERNEL_H

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "kernel/gpu/cuda_impl/cuda_ops/iou_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class IOUGpuKernelMod : public NativeGpuKernelMod {
 public:
  IOUGpuKernelMod() {}
  ~IOUGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                    void *stream_ptr);

  using IOULaunchFunc = std::function<bool(IOUGpuKernelMod *, const std::vector<KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  IOULaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, IOULaunchFunc>> func_list_;

  size_t anchor_boxes_len_{0};
  size_t gt_boxes_len_{0};
  enum input_list_ { ANCHOR_BOXES, GT_BOXES };
  enum output_list_ { IOU_VALUE };
  enum iou_mod_ { IOU_MODE, IOF_MODE };
  int mode_{IOU_MODE};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_IOU_GPU_KERNEL_H
