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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GRAD_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/broadcast_grad_grad_impl.cuh"

namespace mindspore {
namespace kernel {
class BroadcastOpGradGradGpuKernelMod : public NativeGpuKernelMod,
                                        public MatchKernelHelper<BroadcastOpGradGradGpuKernelMod> {
 public:
  BroadcastOpGradGradGpuKernelMod() = default;
  ~BroadcastOpGradGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;
  using KernelFunc = std::vector<std::pair<KernelAttr, BroadcastOpGradGradGpuKernelMod::KernelRunFunc>>;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs);

 private:
  bool GetOpType();

  BroadcastGradGradOpType op_type_{BROADCAST_GRAD_GRAD_TYPE_INVALID};
  size_t output_num_{1};
  bool need_broadcast_{false};
  bool is_null_input_{false};
  bool grad_x_{false};
  bool grad_y_{false};
  std::vector<size_t> x1_shape_;
  std::vector<size_t> x2_shape_;
  std::vector<size_t> sopd_grad_shape_;
  cudaStream_t cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GRAD_GPU_KERNEL_H_
