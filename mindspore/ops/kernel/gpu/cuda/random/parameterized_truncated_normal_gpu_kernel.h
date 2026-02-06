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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_PARAMETERIZED_TRUNCATED_NORMAL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_PARAMETERIZED_TRUNCATED_NORMAL_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
class ParameterizedTruncatedNormalGpuKernelMod : public NativeGpuKernelMod,
                                                 public MatchKernelHelper<ParameterizedTruncatedNormalGpuKernelMod> {
 public:
  ParameterizedTruncatedNormalGpuKernelMod() = default;
  ~ParameterizedTruncatedNormalGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  void ResetResource() noexcept {
    output_elements_ = 0;
    is_null_input_ = false;
    batch_size_ = 0;
    samples_per_batch_ = 0;
    output_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  bool is_null_input_{false};
  bool scalar_mean_;
  bool scalar_stdevs_;
  bool scalar_min_;
  bool scalar_max_;
  void *cuda_stream_{nullptr};
  uint32_t device_id_{0};
  uint64_t seed_{0};
  uint64_t seed_offset_{0};
  int64_t unit_output_size_;
  int64_t output_elements_;
  int64_t stdevs_elements_;
  int64_t batch_size_;
  int64_t samples_per_batch_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_PARAMETERIZED_TRUNCATED_NORMAL_GPU_KERNEL_H_
