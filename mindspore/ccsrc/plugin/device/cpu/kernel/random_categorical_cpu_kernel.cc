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

#include "plugin/device/cpu/kernel/random_categorical_cpu_kernel.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void RandomCategoricalCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

  seed_ = static_cast<int>(GetValue<int64_t>(common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed")));
  if (seed_ <= 0) {
    std::random_device rd;
    seed_ = static_cast<int64_t>(rd());
  }
}

void GetCdf(const float *logits_addr, float *dev_cdf, const size_t batch_size, const size_t num_classes) {
  if (num_classes == 0) {
    MS_LOG(EXCEPTION) << "num_classes must > 0";
  }
  size_t size = num_classes * batch_size;
  for (size_t pos = 0; pos < size; pos += num_classes) {
    size_t cur_row = pos / num_classes;
    size_t cur_col = pos % num_classes;
    if (cur_col != 0) {
      return;
    }
    float max_of_row = logits_addr[pos];
    for (size_t i = 1; i < num_classes; i++) {
      if (logits_addr[pos + i] > max_of_row) {
        max_of_row = logits_addr[pos + i];
      }
    }
    dev_cdf[cur_row * num_classes] = std::exp(static_cast<float>(logits_addr[pos] - max_of_row));
    for (size_t i = 1; i < num_classes; i++) {
      float tmp = std::exp(static_cast<float>(logits_addr[pos + i] - max_of_row));
      dev_cdf[cur_row * num_classes + i] = dev_cdf[cur_row * num_classes + i - 1] + tmp;
    }
  }
}

void RandomCategorical(const size_t num_samples, const float *dev_rand, const float *dev_cdf, const size_t batch_size,
                       const size_t num_classes, int32_t *output_addr) {
  size_t size = num_samples * batch_size;
  for (size_t pos = 0; pos < size; pos++) {
    size_t cur_row = pos / num_samples;
    size_t cur_col = pos % num_samples;
    const float to_find = dev_cdf[cur_row * num_classes + num_classes - 1] * dev_rand[cur_row * num_samples + cur_col];

    size_t idx = 0;
    while (dev_cdf[cur_row * num_classes + idx] < to_find) {
      idx++;
    }
    output_addr[pos] = static_cast<int32_t>(idx);
  }
}

bool RandomCategoricalCpuKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kSizeThree) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << inputs.size()
                      << "input(s).";
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << outputs.size()
                      << "output(s).";
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(outputs[0]);

  float *input_tensor = reinterpret_cast<float *>(inputs[0]->addr);
  int num_sample = reinterpret_cast<int *>(inputs[1]->addr)[0];
  int *output = reinterpret_cast<int *>(outputs[0]->addr);

  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(output);

  int batch_size = input_shape_[0];
  int num_classes = input_shape_[input_shape_.size() - 1];

  std::vector<float> host_cdf(batch_size * num_classes);
  GetCdf(input_tensor, host_cdf.data(), batch_size, num_classes);
  std::uniform_real_distribution<> dist(0, 1);
  rng_.seed(seed_);

  std::vector<float> host_rand(batch_size * num_sample);
  for (int j = 0; j < num_sample; j++) {
    float random = dist(rng_);
    for (int i = 0; i < batch_size; ++i) {
      host_rand[i * num_sample + j] = random;
    }
  }
  RandomCategorical(num_sample, host_rand.data(), host_cdf.data(), batch_size, num_classes, output);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomCategorical, RandomCategoricalCpuKernel);
}  // namespace kernel
}  // namespace mindspore
