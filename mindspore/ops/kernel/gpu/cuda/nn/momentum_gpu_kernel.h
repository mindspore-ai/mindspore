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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include "mindspore/ops/infer/apply_momentum.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 5;
class MomentumGpuKernelMod : public NativeGpuKernelMod {
 public:
  MomentumGpuKernelMod() = default;
  ~MomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *stream_ptr) override {
    launch_func_(this, inputs, stream_ptr);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (inputs.size() != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << INPUT_NUM << ", but got "
                        << inputs.size();
    }
    use_nesterov_ = GetValue<bool>(primitive_->GetAttr(ops::kUseNesterov));

    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    }
    launch_func_ = func_list_[index].second;
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return KernelMod::Resize(inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool use_nesterov_{false};

  template <typename T, typename S, typename G>
  void LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, void *stream_ptr);
  using LaunchFunc =
    std::function<void(MomentumGpuKernelMod *, const std::vector<kernel::KernelTensor *> &, void *stream_ptr)>;
  LaunchFunc launch_func_;

  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_
