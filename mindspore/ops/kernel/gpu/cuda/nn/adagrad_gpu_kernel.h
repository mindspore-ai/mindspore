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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_ADAGRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_ADAGRAD_GPU_KERNEL_H

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/adagrad_impl.cuh"

namespace mindspore {
namespace kernel {
class AdagradGpuKernelMod : public NativeGpuKernelMod {
 public:
  AdagradGpuKernelMod()
      : variable_size_(0),
        accumulation_size_(0),
        learning_rate_size_(0),
        gradient_size_(0),
        variable_shape_(0),
        accumulation_shape_(0),
        gradient_shape_(0),
        update_slots(true),
        kernel_name_("ApplyAdagrad") {}

  ~AdagradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using AdagradLaunchFunc = std::function<bool(AdagradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  AdagradLaunchFunc kernel_func_{nullptr};
  static std::vector<std::pair<KernelAttr, AdagradLaunchFunc>> func_list_;
  size_t variable_size_;
  size_t accumulation_size_;
  size_t learning_rate_size_;
  size_t gradient_size_;
  size_t variable_shape_;
  size_t accumulation_shape_;
  size_t gradient_shape_;
  bool update_slots;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_ADAGRAD_GPU_KERNEL_H
