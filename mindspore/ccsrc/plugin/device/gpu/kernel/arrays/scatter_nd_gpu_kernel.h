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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_nd.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ScatterNdGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ScatterNdGpuKernelMod> {
 public:
  ScatterNdGpuKernelMod() {}
  ~ScatterNdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void CalSize(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);

  void *stream_ptr_{nullptr};
  std::vector<int64_t> attr_shape_;
  std::vector<size_t> vec_indices_stride_;

  static constexpr size_t kShapeIndex_{2};

  size_t block_size_{1};
  size_t indices_dim_0_{0};
  size_t indices_dim_1_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H
