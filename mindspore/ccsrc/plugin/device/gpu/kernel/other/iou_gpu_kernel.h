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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/iou_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class IOUGpuKernelMod : public NativeGpuKernelMod {
 public:
  IOUGpuKernelMod() {}
  ~IOUGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using IOULaunchFunc = std::function<bool(IOUGpuKernelMod *, const std::vector<AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  IOULaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, IOULaunchFunc>> func_list_;

  size_t anchor_boxes_len_{0};
  size_t gt_boxes_len_{0};
  bool is_null_input_{false};
  enum input_list_ { ANCHOR_BOXES, GT_BOXES };
  enum output_list_ { IOU_VALUE };
  enum iou_mod_ { IOU_MODE, IOF_MODE };
  int mode_{IOU_MODE};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_IOU_GPU_KERNEL_H
