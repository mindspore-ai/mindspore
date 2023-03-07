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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DISCOUNTED_RETURN_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DISCOUNTED_RETURN_KERNEL_H_

#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/cuda_impl/rl/discounted_return_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kGammaAttrName = "gamma";
constexpr size_t kInputNum = 3;

template <typename T>
class DiscountedReturnGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  DiscountedReturnGpuKernelMod() = default;
  ~DiscountedReturnGpuKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    gamma_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, kGammaAttrName);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be " << kInputNum << ", but got "
                        << input_num;
    }

    const std::vector<int64_t> &reward_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    const std::vector<int64_t> &done_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    if (reward_shape.size() == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of reward cannot be 0, but got "
                        << reward_shape.size();
    }

    // Reshape reward to [timestep, env, else], done to [timestep, env], last_value to [env, else].
    timestep_ = LongToInt(reward_shape[0]);
    for (size_t i = 1; i < done_shape.size(); i++) {
      env_num_ *= i;
    }

    int total_elements = 1;
    for (size_t j = 0; j < reward_shape.size(); j++) {
      total_elements *= LongToInt(reward_shape[j]);
    }

    MS_EXCEPTION_IF_ZERO("timestep", timestep_);
    MS_EXCEPTION_IF_ZERO("env_num", env_num_);
    element_per_env_ = total_elements / timestep_ / env_num_;

    input_size_list_.push_back(total_elements * sizeof(T));
    input_size_list_.push_back(timestep_ * env_num_ * sizeof(bool));
    input_size_list_.push_back(env_num_ * element_per_env_ * sizeof(T));
    output_size_list_.push_back(total_elements * sizeof(T));
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    T *reward = GetDeviceAddress<T>(inputs, 0);
    bool *done = GetDeviceAddress<bool>(inputs, 1);
    T *last_value = GetDeviceAddress<T>(inputs, 2);
    T *result = GetDeviceAddress<T>(outputs, 0);

    DiscountedReturn(timestep_, env_num_, element_per_env_, gamma_, reward, done, last_value, result,
                     reinterpret_cast<cudaStream_t>(stream));
    return true;
  }

  void InitSizeLists() override{};

 private:
  float gamma_ = 0.99;
  int timestep_ = 1;
  int env_num_ = 1;
  int element_per_env_ = 1;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DISCOUNTED_RETURN_KERNEL_H_
