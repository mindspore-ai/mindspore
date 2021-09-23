/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <random>
#include <thread>
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/cpu/random_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUniformIntInputsNum = 3;
constexpr size_t kUniformRealInputsNum = 1;
constexpr size_t kUniformIntOutputsNum = 1;
constexpr size_t kUniformRealOutputsNum = 1;
constexpr size_t kStandardNormalOutputsNum = 1;
}  // namespace
void StandardNormal(float *output, std::normal_distribution<float> distribution,
                    std::default_random_engine random_generator, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = distribution(random_generator);
  }
}

void LaunchStandardNormal(unsigned int seed, const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  // multithreading
  size_t lens = outputs[0]->size / sizeof(float);
  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = lens < 128 * max_thread_num ? std::ceil(lens / 128.0) : max_thread_num;
  if (thread_num < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num " << thread_num;
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  if (once_compute_size < 1) {
    MS_LOG(ERROR) << "Invalid value: once_compute_size " << once_compute_size;
    return;
  }
  std::normal_distribution<float> distribution;
  while (start < lens) {
    // avoid different threads using the same seed to generate the same random number
    std::default_random_engine random_generator(++seed);
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    (void)threads.emplace_back(std::thread(StandardNormal, output, distribution, random_generator, start, end));
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void LaunchUniformInt(unsigned int seed, const std::vector<AddressPtr> &inputs,
                      const std::vector<AddressPtr> &outputs) {
  // Init min/max values.
  int min_val = reinterpret_cast<int *>(inputs[1]->addr)[0];
  int max_val = reinterpret_cast<int *>(inputs[2]->addr)[0];
  if (max_val <= min_val) {
    MS_LOG(EXCEPTION) << "Invalid min/max values: (" << min_val << "/" << max_val << ")";
  }

  // Init output address.
  auto output = reinterpret_cast<int *>(outputs[0]->addr);

  // Init sample number.
  size_t num_sample = outputs[0]->size / sizeof(int);

  // Init random int generator.
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> distrib(min_val, max_val - 1);

  // Generate random int values.
  for (size_t i = 0; i < num_sample; ++i) {
    output[i] = distrib(gen);
  }
}

void LaunchUniformReal(unsigned int seed, const std::vector<AddressPtr> &inputs,
                       const std::vector<AddressPtr> &outputs) {
  // Init output address.
  auto output = reinterpret_cast<float *>(outputs[0]->addr);

  // Init sample number.
  size_t num_sample = outputs[0]->size / sizeof(int);

  // Init random real generator.
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  // Generate random real values.
  for (size_t i = 0; i < num_sample; ++i) {
    output[i] = distrib(gen);
  }
}

void RandomCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto iter = kRandomOpTypeMap.find(kernel_name_);
  if (iter == kRandomOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Random operation " << kernel_name_ << " is not supported.";
  } else {
    random_op_type_ = iter->second;
  }
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  seed_ = LongToInt(GetValue<int64_t>(prim->GetAttr("seed")));
  seed2_ = LongToInt(GetValue<int64_t>(prim->GetAttr("seed2")));
}

bool RandomCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                             const std::vector<kernel::AddressPtr> &outputs) {
  unsigned int RNG_seed = 0;
  std::random_device rd;
  if (seed2_ != 0) {
    RNG_seed = IntToUint(seed2_);
  } else if (seed_ != 0) {
    RNG_seed = IntToUint(seed_);
  } else {
    RNG_seed = rd();
  }

  if (random_op_type_ == RANDOM_OP_NORMAL) {
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStandardNormalOutputsNum, kernel_name_);
    LaunchStandardNormal(RNG_seed, outputs);
  } else if (random_op_type_ == RANDOM_OP_UNIFORM_INT) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniformIntInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniformIntOutputsNum, kernel_name_);
    LaunchUniformInt(RNG_seed, inputs, outputs);
  } else if (random_op_type_ == RANDOM_OP_UNIFORM_REAL) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniformRealInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniformRealOutputsNum, kernel_name_);
    LaunchUniformReal(RNG_seed, inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Random operation " << random_op_type_ << " is not supported.";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
