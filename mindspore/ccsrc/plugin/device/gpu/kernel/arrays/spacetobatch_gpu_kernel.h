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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "mindspore/core/ops/space_to_batch.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/spacetobatch_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t PADDING_SHAPE_0 = 2;
constexpr size_t PADDING_SHAPE_1 = 2;

class SpaceToBatchGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<SpaceToBatchGpuKernelMod> {
 public:
  SpaceToBatchGpuKernelMod() = default;
  ~SpaceToBatchGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  void *stream_ptr_{nullptr};
  std::vector<std::vector<int64_t>> paddings_;
  std::vector<size_t> input_shape_;
  int64_t block_size_{0};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t in_{0};
  size_t ic_{0};
  size_t ih_{0};
  size_t iw_{0};
  size_t on_{0};
  size_t oc_{0};
  size_t oh_{0};
  size_t ow_{0};
  size_t unit_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_
