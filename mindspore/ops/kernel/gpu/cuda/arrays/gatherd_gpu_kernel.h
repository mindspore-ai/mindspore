/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHERD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHERD_GPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/gatherd.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"

namespace mindspore {
namespace kernel {
class GatherDGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherDGpuKernelMod() {}
  ~GatherDGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (!kernel_func_) {
      MS_LOG(ERROR) << "GatherFwdGpu's kernel function is not initialized.";
      return false;
    }
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using GatherFwdFunc = std::function<bool(GatherDGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &, void *)>;

  GatherFwdFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, GatherFwdFunc>> func_list_;

  bool is_null_input_{false};
  ShapeHelper input_shape_helper_;
  ShapeHelper index_shape_helper_;
  size_t dim_{0};
  size_t index_num_{0};
  size_t rank_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHERD_GPU_KERNEL_H_
