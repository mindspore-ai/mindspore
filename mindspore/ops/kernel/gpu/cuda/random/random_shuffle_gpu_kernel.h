/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_SHUFFLE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_SHUFFLE_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "mindspore/ops/infer/random_shuffle.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/random_op_impl.cuh"

namespace mindspore {
namespace kernel {
class RandomShuffleGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomShuffleGpuKernelMod() = default;
  ~RandomShuffleGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  std::vector<int> GetShuffleIndex();

  using RandomShuffleFunc =
    std::function<bool(RandomShuffleGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, RandomShuffleFunc>> func_list_;
  RandomShuffleFunc kernel_func_;

  int64_t outer_size_{1};
  int64_t inner_size_{1};
  size_t shuffle_size_{1};
  size_t batch_rank_{0};
  uint64_t seed_{0};
  uint64_t seed_offset_{0};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  std::vector<int64_t> input_shape_;
  std::default_random_engine generator_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_SHUFFLE_GPU_KERNEL_H_
