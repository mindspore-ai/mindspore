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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_V2_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_V2_GRAD_GPU_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/ctcloss_v2_impl.cuh"

namespace mindspore {
namespace kernel {
class CTCLossV2GradGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<CTCLossV2GradGpuKernelMod> {
 public:
  CTCLossV2GradGpuKernelMod() = default;
  ~CTCLossV2GradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  [[nodiscard]] const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  // Variables for the operator itself
  int64_t blank_{0};
  // Stands for T
  int64_t time_series_{0};
  // Stands for N
  int64_t batch_size_{0};
  // Stands for C
  int64_t num_labels_{0};
  // Stands for S
  int64_t max_target_length_{0};

  dim3 log_probs_shape_;
  dim3 log_alpha_shape_;

  bool zero_infinity_ = false;

  bool is_null_input_{false};
  cudaStream_t stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_V2_GRAD_GPU_KERNEL_H_
