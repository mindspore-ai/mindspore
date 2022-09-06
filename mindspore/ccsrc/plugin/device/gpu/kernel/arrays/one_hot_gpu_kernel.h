/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/one_hot_impl.cuh"

namespace mindspore {
namespace kernel {
class OneHotGpuKernelMod : public NativeGpuKernelMod {
 public:
  OneHotGpuKernelMod() = default;
  ~OneHotGpuKernelMod() = default;

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
  template <typename T, typename S, typename G = int>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using OneHotLaunchFunc =
    std::function<bool(OneHotGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;

  static std::vector<std::pair<KernelAttr, OneHotLaunchFunc>> func_list_;
  OneHotLaunchFunc kernel_func_;
  bool is_dynamic_shape_ = false;
  size_t depth_{0};
  size_t left_dim_size_{1};
  size_t right_dim_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H
