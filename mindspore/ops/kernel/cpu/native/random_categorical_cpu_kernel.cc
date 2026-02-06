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

#include "kernel/cpu/native/random_categorical_cpu_kernel.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <utility>

#include "infer/random_categorical.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
namespace random_categorical_cpu {
namespace {
using KernelRunFunc = RandomCategoricalCpuKernel::KernelRunFunc;
#define RANDOM_CATEGORICAL_ADD_KERNEL(logits_dtype, nun_sample_dtype, seed_dtype, output_dtype, logits_type, \
                                      output_type)                                                           \
  {KernelAttr()                                                                                              \
     .AddInputAttr(kNumberType##logits_dtype)                                                                \
     .AddInputAttr(kNumberType##nun_sample_dtype)                                                            \
     .AddInputAttr(kNumberType##seed_dtype)                                                                  \
     .AddOutputAttr(kNumberType##output_dtype),                                                              \
   &RandomCategoricalCpuKernel::LaunchKernel<logits_type, output_type>}
}  // namespace

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &RandomCategoricalCpuKernel::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, RandomCategoricalCpuKernel::KernelRunFunc>> func_list = {
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int32, Int32, Int16, float16, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int32, Int32, Int32, float16, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int32, Int32, Int64, float16, int64_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int32, Int32, Int16, float, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int32, Int32, Int32, float, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int32, Int32, Int64, float, int64_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int32, Int32, Int16, double, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int32, Int32, Int32, double, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int32, Int32, Int64, double, int64_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int64, Int64, Int16, float16, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int64, Int64, Int32, float16, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float16, Int64, Int64, Int64, float16, int64_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int64, Int64, Int16, float, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int64, Int64, Int32, float, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float32, Int64, Int64, Int64, float, int64_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int64, Int64, Int16, double, int16_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int64, Int64, Int32, double, int32_t),
    RANDOM_CATEGORICAL_ADD_KERNEL(Float64, Int64, Int64, Int64, double, int64_t)};
  return func_list;
}

bool RandomCategoricalCpuKernel::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int RandomCategoricalCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = NativeCpuKernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(0)->GetShapeVector();
  return KRET_OK;
}

template <typename T>
void GetCdf(T *logits_addr, T *dev_cdf, const size_t batch_size, const size_t num_classes) {
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
    T max_of_row = logits_addr[pos];
    for (size_t i = 1; i < num_classes; i++) {
      if (logits_addr[pos + i] > max_of_row) {
        max_of_row = logits_addr[pos + i];
      }
    }
    float exp1 = static_cast<float>(logits_addr[pos] - max_of_row);
    dev_cdf[cur_row * num_classes] = static_cast<T>(std::exp(exp1));
    for (size_t i = 1; i < num_classes; i++) {
      float exp2 = static_cast<float>(logits_addr[pos + i] - max_of_row);
      T tmp = static_cast<T>(std::exp(exp2));
      dev_cdf[cur_row * num_classes + i] = dev_cdf[cur_row * num_classes + i - 1] + tmp;
    }
  }
}

template <typename T1, typename T2>
void RandomCategoricalFunc(const size_t num_samples, const T1 *dev_rand, const T1 *dev_cdf, const size_t batch_size,
                           const size_t num_classes, T2 *output_addr) {
  size_t size = num_samples * batch_size;
  for (size_t pos = 0; pos < size; pos++) {
    size_t cur_row = pos / num_samples;
    size_t cur_col = pos % num_samples;
    const T1 to_find = dev_cdf[cur_row * num_classes + num_classes - 1] * dev_rand[cur_row * num_samples + cur_col];

    size_t idx = 0;
    while (dev_cdf[cur_row * num_classes + idx] < to_find) {
      idx++;
    }
    output_addr[pos] = static_cast<T2>(idx);
  }
}

template <typename T1, typename T2>
bool RandomCategoricalCpuKernel::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kSizeThree = 3;
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

  T1 *input_tensor = GetDeviceAddress<T1>(inputs, kIndex0);
  int num_sample = GetDeviceAddress<int>(inputs, kIndex1)[0];
  int input_seed = GetDeviceAddress<int>(inputs, kIndex2)[0];
  T2 *output = GetDeviceAddress<T2>(outputs, kIndex0);

  int batch_size = input_shape_[0];
  int num_classes = input_shape_[input_shape_.size() - 1];

  std::vector<T1> host_cdf(batch_size * num_classes);
  GetCdf(input_tensor, host_cdf.data(), batch_size, num_classes);
  std::uniform_real_distribution<> dist(0, 1);
  // Is the op running for the first time or multiple times but the seed has changed
  if (init_state_ || input_seed != init_seed_) {
    if (init_state_) {
      init_state_ = false;
    }
    init_seed_ = input_seed;
    uint64_t seed = random::GetSeed(0, static_cast<uint64_t>(input_seed));
    rng_.seed(seed);
  }

  std::vector<T1> host_rand(batch_size * num_sample);
  for (int j = 0; j < num_sample; j++) {
    float random = dist(rng_);
    for (int i = 0; i < batch_size; ++i) {
      host_rand[i * num_sample + j] = static_cast<T1>(random);
    }
  }
  RandomCategoricalFunc(num_sample, host_rand.data(), host_cdf.data(), batch_size, num_classes, output);
  return true;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomCategorical, RandomCategoricalCpuKernel);
}  // namespace random_categorical_cpu
}  // namespace kernel
}  // namespace mindspore
