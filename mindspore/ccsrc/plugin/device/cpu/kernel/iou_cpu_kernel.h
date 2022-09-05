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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_IOU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_IOU_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class IOUCpuKernelMod : public NativeCpuKernelMod {
 public:
  IOUCpuKernelMod() = default;
  ~IOUCpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  using IOULaunchFunc =
    std::function<bool(IOUCpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

 private:
  static std::vector<std::pair<KernelAttr, IOULaunchFunc>> func_list_;
  IOULaunchFunc kernel_func_;
  size_t anchor_boxes_size_{0};
  size_t gt_boxes_size_{0};
  size_t iou_size_{0};
  enum input_list_ { ANCHOR_BOXES, GT_BOXES };
  enum output_list_ { IOU_VALUE };
  enum iou_mod_ { IOU_MODE, IOF_MODE };
  int mode_{IOU_MODE};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_IOU_CPU_KERNEL_H_
