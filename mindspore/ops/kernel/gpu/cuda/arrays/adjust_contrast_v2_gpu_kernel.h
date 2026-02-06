/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAY_ADJUST_CONTRAST_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAY_ADJUST_CONTRAST_V2_GPU_KERNEL_H_

#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/kernel_constants.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class AdjustContrastV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  AdjustContrastV2GpuKernelMod() {
    KernelMod::kernel_name_ = "AdjustContrastv2";
    total_ = 0;
    per_batch_elements_ = 0;
    is_null_input_ = false;
    data_unit_size_ = 0;
    stream_ptr_ = nullptr;
  }
  ~AdjustContrastV2GpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using AdjustContrastv2Func = std::function<bool(
    AdjustContrastV2GpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;
  AdjustContrastv2Func kernel_func_;
  static std::vector<std::pair<KernelAttr, AdjustContrastv2Func>> func_list_;

 private:
  void ResetResource();
  void InitSizeLists();
  void *stream_ptr_;
  int total_;
  int per_batch_elements_;
  bool is_null_input_;
  // default values
  size_t data_unit_size_; /* size of T */
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAY_ADJUST_CONTRAST_V2_GPU_KERNEL_H_
