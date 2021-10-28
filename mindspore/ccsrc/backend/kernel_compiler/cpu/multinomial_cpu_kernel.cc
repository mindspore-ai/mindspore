/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/multinomial_cpu_kernel.h"
#include <algorithm>
#include <random>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void MultinomialCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  // The dimensions of input tensor must be 1 or 2, with data type of float32.
  if (input_shape_.size() == 1) {
    workspace_size_list_.push_back(input_shape_[0] * sizeof(float));
  } else if (input_shape_.size() == 2) {
    workspace_size_list_.push_back(input_shape_[1] * sizeof(float));
  }

  seed_ = static_cast<int>(GetValue<int64_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed")));
  seed2_ = static_cast<int>(GetValue<int64_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed2")));
}

bool MultinomialCpuKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &workspace,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "Invalid input numbers, expect input number 2, but actual input number " << inputs.size();
  }
  if (workspace.size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid workspace numbers, expect workspace number 1, actual workspace number "
                      << workspace.size();
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid output numbers, expect output number 1, actual output number " << outputs.size();
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(workspace[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);

  float *input_tensor = reinterpret_cast<float *>(inputs[0]->addr);
  int num_sample = reinterpret_cast<int *>(inputs[1]->addr)[0];
  int *output = reinterpret_cast<int *>(outputs[0]->addr);
  float *cumulative_value = reinterpret_cast<float *>(workspace[0]->addr);

  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(cumulative_value);

  int num_row = 1;
  if (input_shape_.size() == 2) {
    num_row = input_shape_[0];
  }
  int num_col = input_shape_[input_shape_.size() - 1];

  for (int i = 0; i < num_row; ++i) {
    // Compute the cumulative array.
    cumulative_value[i * num_col] = input_tensor[i * num_col];
    for (int j = 1; j < num_col; ++j) {
      size_t index = i * num_col + j;
      cumulative_value[index] = cumulative_value[index - 1] + input_tensor[index];
    }

    // Normalize the cumulative array.
    float sum = cumulative_value[(i + 1) * num_col - 1];
    if (sum != 0) {
      for (int k = 0; k < num_col; ++k) {
        size_t index = i * num_col + k;
        cumulative_value[index] /= sum;
      }
    }

    // Initialize random generator.
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    int64_t RNG_seed = 0;
    if (seed2_ > 0) {
      RNG_seed = seed2_;
    } else if (seed_ > 0) {
      RNG_seed = seed_;
    } else {
      std::random_device rd;
      RNG_seed = static_cast<int64_t>(rd());
    }
    std::default_random_engine rng{RNG_seed};

    // Sample data from cumulative array.
    for (int n = 0; n < num_sample; ++n) {
      auto rand_prob = dist(rng);
      int begin = 0;
      int end = num_col - 1;

      while (end - begin > 0) {
        int pivot = begin + (end - begin) / 2;
        float pivot_prob = cumulative_value[i * num_col + pivot];
        if (pivot_prob > rand_prob) {
          end = pivot;
        } else {
          begin = pivot + 1;
        }
      }
      output[i * num_sample + n] = begin;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
