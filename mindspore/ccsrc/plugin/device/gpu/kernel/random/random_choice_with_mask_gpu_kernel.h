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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_

#include <vector>
#include <chrono>
#include <random>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_choice_with_mask_impl.cuh"

namespace mindspore {
namespace kernel {
class RandomChoiceWithMaskGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomChoiceWithMaskGpuKernelMod() = default;
  ~RandomChoiceWithMaskGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  void InitWorkSpaceSizeLists() {
    if (count_ > kSmallK || input_shape_size_ > 1) {
      workspace_size_list_.push_back(input_size_ * input_shape_size_ * sizeof(int32_t));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(int32_t));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(int32_t));
      int blocknum = std::ceil(static_cast<float>(ceil_power2_) / BLOCKSIZE);
      workspace_size_list_.push_back(blocknum * sizeof(int32_t));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(int32_t));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(curandState));
    }
  }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using RandomChoiceWithMaskLaunchFunc =
    std::function<bool(RandomChoiceWithMaskGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, RandomChoiceWithMaskLaunchFunc>> func_list_;
  RandomChoiceWithMaskLaunchFunc kernel_func_;

 private:
  const int kSmallK = 2048;
  int input_shape_size_{0};
  int seed_{0};
  int seed2_{0};
  int input_size_{1};
  int count_{0};
  int ceil_power2_{0};
  std::mt19937 generator_;
  std::vector<int> input_shape_5D_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
