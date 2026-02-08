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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_MULTI_MARGIN_LOSS_Grad_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_MULTI_MARGIN_LOSS_Grad_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr int64_t p_num_1 = 1;
constexpr int64_t p_num_2 = 2;
constexpr int64_t reduction_num_0 = 0;
constexpr int64_t reduction_num_1 = 1;
constexpr int64_t reduction_num_2 = 2;
class MultiMarginLossGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  MultiMarginLossGradGpuKernelMod() {}
  ~MultiMarginLossGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using MultiMarginLossGradFunc =
    std::function<bool(MultiMarginLossGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  int64_t p_;
  float margin_;
  int64_t reduction_;
  size_t unit_size_{1};
  size_t input_elements_{};
  int nframe_;
  int dim_;
  bool has_weight_ = false;
  void *cuda_stream_{nullptr};
  MultiMarginLossGradFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, MultiMarginLossGradFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_MULTI_MARGIN_LOSS_Grad_GPU_KERNEL_H_
