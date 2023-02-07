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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl_opt.cuh"
namespace mindspore {
namespace kernel {
class TransposeGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<TransposeGpuKernelMod> {
 public:
  TransposeGpuKernelMod() = default;
  ~TransposeGpuKernelMod() override = default;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

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

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void GetPermValue(const std::vector<int64_t> &perm);

  void *stream_ptr_{nullptr};
  std::vector<int64_t> input_shape_;
  std::vector<size_t> input_perm_;

  size_t shape_size_{0};
  size_t workspace_size_{0};
  bool is_null_input_;
  bool is_dynamic_perm_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_
