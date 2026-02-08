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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_INDEX_ADD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_INDEX_ADD_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class IndexAddGpuKernelMod : public NativeGpuKernelMod {
 public:
  IndexAddGpuKernelMod() = default;
  ~IndexAddGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool CheckParams();
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using IndexAddFunc =
    std::function<bool(IndexAddGpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, IndexAddFunc>> func_list_;
  IndexAddFunc kernel_func_;

  std::vector<int64_t> x_shape_;
  std::vector<int64_t> index_shape_;
  std::vector<int64_t> y_shape_;
  size_t t_size_{0};  // Equal to sizeof(T).
  int64_t axis_value_{0};
  size_t axis_{0};
  size_t x_size_{0};
  size_t inner_size_{0};
  size_t outer_size_{0};
  size_t x_axis_size_{0};
  size_t y_axis_size_{0};
  bool use_lock_{true};
  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_INDEX_ADD_CPU_KERNEL_H_
