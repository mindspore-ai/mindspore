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
#include "plugin/device/cpu/kernel/random_cpu_kernel.h"
#include <random>
#include <thread>
#include <memory>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUniformIntInputsNum = 3;
constexpr size_t kUniformRealInputsNum = 1;
constexpr size_t kUniformIntOutputsNum = 1;
constexpr size_t kUniformRealOutputsNum = 1;
constexpr size_t kStandardNormalOutputsNum = 1;
constexpr float kRandomBlockSize = 128.0;
constexpr char kKernelName[] = "Random";
}  // namespace
void StandardNormal(float *output, std::normal_distribution<float> distribution,
                    std::default_random_engine random_generator, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = distribution(random_generator);
  }
}

void LaunchStandardNormal(RandomCpuKernelMod *content, unsigned int seed, const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  // multithreading
  size_t lens = outputs[0]->size / sizeof(float);
  auto thread_pool = GetActorMgrInnerThreadPool();
  size_t max_thread_num = thread_pool->GetKernelThreadNum();
  size_t thread_num = lens < kRandomBlockSize * max_thread_num ? std::ceil(lens / kRandomBlockSize) : max_thread_num;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  std::normal_distribution<float> distribution;
  auto task = [&](size_t start, size_t end) {
    auto task_id = start / once_compute_size;
    std::default_random_engine random_generator(seed + task_id);
    StandardNormal(output, distribution, random_generator, start, end);
  };
  ParallelLaunch(task, lens, kRandomBlockSize, content);
}

void LaunchUniformInt(unsigned int seed, const std::vector<AddressPtr> &inputs,
                      const std::vector<AddressPtr> &outputs) {
  // Init min/max values.
  int min_val = reinterpret_cast<int *>(inputs[1]->addr)[0];
  int max_val = reinterpret_cast<int *>(inputs[2]->addr)[0];
  if (max_val <= min_val) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', invalid min/max values: (" << min_val << "/" << max_val << ")";
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

void LaunchUniformReal(unsigned int seed, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &outputs) {
  // Init output address.
  auto output = reinterpret_cast<float *>(outputs[0]->addr);

  // Init sample number.
  size_t num_sample = outputs[0]->size / sizeof(int);

  // Init random real generator.
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  // Generate random real values.
  for (size_t i = 0; i < num_sample; ++i) {
    output[i] = static_cast<float>(distrib(gen));
  }
}

bool RandomCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto iter = kRandomOpTypeMap.find(kernel_type_);
  if (iter == kRandomOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_type_
                      << ", only support these types: StandardNormal, UniformInt or UniformReal currently, but got "
                      << kernel_type_;
  } else {
    random_op_type_ = iter->second;
  }
  seed_ = LongToInt(GetValue<int64_t>(base_operator->GetAttr("seed")));
  seed2_ = LongToInt(GetValue<int64_t>(base_operator->GetAttr("seed2")));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto res = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!res.first) {
    MS_LOG(ERROR) << "For '" << kernel_type_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int RandomCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

bool RandomCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
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
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStandardNormalOutputsNum, kernel_type_);
    LaunchStandardNormal(this, RNG_seed, outputs);
  } else if (random_op_type_ == RANDOM_OP_UNIFORM_INT) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniformIntInputsNum, kernel_type_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniformIntOutputsNum, kernel_type_);
    LaunchUniformInt(RNG_seed, inputs, outputs);
  } else if (random_op_type_ == RANDOM_OP_UNIFORM_REAL) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniformRealInputsNum, kernel_type_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniformRealOutputsNum, kernel_type_);
    LaunchUniformReal(RNG_seed, inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_type_
                      << ", only support these types: StandardNormal, UniformInt or UniformReal currently, but got "
                      << random_op_type_;
  }
  return true;
}

std::vector<KernelAttr> RandomCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kStandardNormal, {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32)}},
    {kUniformInt,
     {KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt32)}},
    {kUniformReal, {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32)}}};
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }
  return iter->second;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, StandardNormal,
                                 []() { return std::make_shared<RandomCpuKernelMod>(kStandardNormal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, UniformInt,
                                 []() { return std::make_shared<RandomCpuKernelMod>(kUniformInt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, UniformReal,
                                 []() { return std::make_shared<RandomCpuKernelMod>(kUniformReal); });
}  // namespace kernel
}  // namespace mindspore
