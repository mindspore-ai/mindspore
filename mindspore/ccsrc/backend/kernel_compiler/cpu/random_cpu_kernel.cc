/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
void StandardNormal(float *output, std::normal_distribution<float> distribution,
                    std::default_random_engine random_generator, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = distribution(random_generator);
  }
}
void LaunchStandardNormal(int seed, int seed2, const std::vector<AddressPtr> &outputs) {
  unsigned int RNG_seed;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = IntToUint(seed2);
  } else if (seed != 0) {
    RNG_seed = IntToUint(seed);
  } else {
    RNG_seed = rd();
  }

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
    std::default_random_engine random_generator(++RNG_seed);
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    threads.emplace_back(std::thread(StandardNormal, output, distribution, random_generator, start, end));
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void RandomCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  auto iter = kRandomOpTypeMap.find(kernel_name);
  if (iter == kRandomOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Random operation " << kernel_name << " is not supported.";
  } else {
    random_op_type_ = iter->second;
  }

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if ((random_op_type_ == RANDOM_OP_NORMAL) && input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but random op needs 1 input.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but random op needs 1 output.";
  }

  seed_ = LongToInt(GetValue<int64_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed")));
  seed2_ = LongToInt(GetValue<int64_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed2")));
}

bool RandomCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  switch (random_op_type_) {
    case RANDOM_OP_NORMAL: {
      LaunchStandardNormal(seed_, seed2_, outputs);
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "Random operation " << random_op_type_ << " is not supported.";
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
