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
class DiscountedReturnGpuKernelMod : public NativeGpuKernelMod {
 public:
  DiscountedReturnGpuKernelMod() = default;
  ~DiscountedReturnGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    size_t input_num = inputs.size();
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputNum << ", but got "
                        << input_num;
    }
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    for (const auto &input : inputs) {
      auto input_shape = input->GetShapeVector();
      if (!IsValidShape(input_shape)) {
        return KRET_UNKNOWN_SHAPE;
      }
    }
    gamma_ = GetValue<float>(primitive_->GetAttr(kGammaAttrName));
    const std::vector<int64_t> &reward_shape = inputs[kIndex0]->GetDeviceShapeVector();
    const std::vector<int64_t> &done_shape = inputs[kIndex1]->GetDeviceShapeVector();
    if (reward_shape.size() == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of reward cannot be 0, but got "
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

    output_size_list_.push_back(total_elements * sizeof(T));
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream) override {
    T *reward = GetDeviceAddress<T>(inputs, 0);
    bool *done = GetDeviceAddress<bool>(inputs, 1);
    T *last_value = GetDeviceAddress<T>(inputs, 2);
    T *result = GetDeviceAddress<T>(outputs, 0);

    auto status = DiscountedReturn(timestep_, env_num_, element_per_env_, gamma_, reward, done, last_value, result,
                                   reinterpret_cast<cudaStream_t>(stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

 private:
  float gamma_ = 0.99;
  int timestep_ = 1;
  int env_num_ = 1;
  int element_per_env_ = 1;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DISCOUNTED_RETURN_KERNEL_H_
