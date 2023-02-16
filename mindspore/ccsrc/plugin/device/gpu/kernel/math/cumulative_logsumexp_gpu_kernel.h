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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUMULATIVELOGSUMEXP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUMULATIVELOGSUMEXP_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cumulativelogsumexp_impl.cuh"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr int kMaxDimsSize = 3;
class CumulativeLogsumexpGpuKernelMod : public NativeGpuKernelMod {
 public:
  CumulativeLogsumexpGpuKernelMod() = default;
  ~CumulativeLogsumexpGpuKernelMod() override = default;

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

 private:
  void Reshape();
  void ResetResource() noexcept;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using CumulativeLogsumexpLaunchFunc =
    std::function<bool(CumulativeLogsumexpGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, CumulativeLogsumexpLaunchFunc>> func_list_;
  CumulativeLogsumexpLaunchFunc kernel_func_;
  int axis_{0};
  bool exclusive_{false};
  bool reverse_{false};
  bool is_null_input_{false};
  size_t stride_{0};
  size_t stride2_{0};
  size_t dims_[kMaxDimsSize] = {};
  std::vector<size_t> shape_{};
  bool is_dynamic_shape_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUMULATIVELOSUMEXP_GPU_KERNEL_H_
