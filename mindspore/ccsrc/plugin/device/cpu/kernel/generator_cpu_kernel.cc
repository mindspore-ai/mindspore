/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "plugin/device/cpu/kernel/generator_cpu_kernel.h"
#include <securec.h>
#include <algorithm>
#include <limits>
#include <random>
#include <functional>
#include <cmath>
#include <string>
#include <unordered_map>
#include "mindspore/core/ops/ops_func_impl/generator.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using namespace ops::generator;
namespace {
constexpr char kGenerator[] = "Generator";
constexpr size_t kOutputSeedIdx = 0;
constexpr size_t kOutputOffsetIdx = 1;
constexpr size_t kOutputStateIdx = 2;
constexpr size_t kOutputNum = 3;

using ComputeFunc =
  std::function<bool(const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

void GeneratorCheck(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs, const size_t input_num, const size_t output_num,
                    const std::string &operation) {
  if (inputs.size() != input_num) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", expected " << input_num << " inputs for '" << operation
                      << "' operation, but got '" << inputs.size() << "'.";
  }
  if (outputs.size() != output_num) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", expected " << output_num << " outputs for '" << operation
                      << "' operation, but got '" << outputs.size() << "'.";
  }
}

bool PrepareOutput(param_type seed, param_type offset, const std::vector<kernel::KernelTensor *> &outputs) {
  auto seed_addr = GetDeviceAddress<param_type>(outputs, kOutputSeedIdx);
  auto offset_addr = GetDeviceAddress<param_type>(outputs, kOutputOffsetIdx);
  MS_EXCEPTION_IF_NULL(seed_addr);
  MS_EXCEPTION_IF_NULL(offset_addr);
  *seed_addr = seed;
  *offset_addr = offset;
  // Calculate State
  auto state_addr = GetDeviceAddress<state_type>(outputs, kOutputStateIdx);
  MS_EXCEPTION_IF_NULL(state_addr);
  const auto param_size = sizeof(param_type);
  const auto output_size = param_size * 2 / sizeof(state_type);
  auto ret = memcpy_s(static_cast<void *>(state_addr), output_size, static_cast<void *>(&seed), param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  ret = memcpy_s(static_cast<void *>(state_addr + param_size), output_size - param_size, static_cast<void *>(&offset),
                 param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  return true;
}

/*
  Input: cmd, seed_param, offset_param, step_size
  Update offset parameter and return the old offset value.
*/
bool StepCompute(const std::vector<kernel::KernelTensor *> &inputs,
                 const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 4;
  constexpr size_t kSeedParamIdx = 1;
  constexpr size_t kOffsetParamIdx = 2;
  constexpr size_t kStepSizeIdx = 3;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "step");

  auto seed_param = GetDeviceAddress<param_type>(inputs, kSeedParamIdx);
  auto offset_param = GetDeviceAddress<param_type>(inputs, kOffsetParamIdx);
  auto step_size = GetDeviceAddress<param_type>(inputs, kStepSizeIdx);
  MS_EXCEPTION_IF_NULL(seed_param);
  MS_EXCEPTION_IF_NULL(offset_param);
  MS_EXCEPTION_IF_NULL(step_size);

  auto old_offset = *offset_param;
  *offset_param = old_offset + *step_size;
  return PrepareOutput(*seed_param, old_offset, outputs);
}

/*
  Input: cmd, seed, offset
  Generate random number as new seed. Reset offset.
*/
bool SeedCompute(const std::vector<kernel::KernelTensor *> &inputs,
                 const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 3;
  constexpr size_t kSeedIdx = 1;
  constexpr size_t kOffsetIdx = 2;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "seed");

  auto seed_param = GetDeviceAddress<param_type>(inputs, kSeedIdx);
  auto offset_param = GetDeviceAddress<param_type>(inputs, kOffsetIdx);
  MS_EXCEPTION_IF_NULL(seed_param);
  MS_EXCEPTION_IF_NULL(offset_param);
  auto rd = static_cast<param_type>(std::random_device()());
  *seed_param = rd;
  *offset_param = 0;
  return PrepareOutput(rd, 0, outputs);
}

/*
  Input: cmd, seed, offset
  Return a tensor representing the generator state.
*/
bool GetStateCompute(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 3;
  constexpr size_t kSeedIdx = 1;
  constexpr size_t kOffsetIdx = 2;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "get_state");

  auto seed = GetDeviceAddress<param_type>(inputs, kSeedIdx);
  auto offset = GetDeviceAddress<param_type>(inputs, kOffsetIdx);
  MS_EXCEPTION_IF_NULL(seed);
  MS_EXCEPTION_IF_NULL(offset);
  return PrepareOutput(*seed, *offset, outputs);
}

/*
  Input: cmd, seed_param, offset_param, state
  Restore seed and offset from state
*/
bool SetStateCompute(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 4;
  constexpr size_t kSeedIdx = 1;
  constexpr size_t kOffsetIdx = 2;
  constexpr size_t kStateIdx = 3;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "set_state");

  auto state = GetDeviceAddress<state_type>(inputs, kStateIdx);
  auto seed_param = GetDeviceAddress<param_type>(inputs, kSeedIdx);
  auto offset_param = GetDeviceAddress<param_type>(inputs, kOffsetIdx);
  MS_EXCEPTION_IF_NULL(state);
  MS_EXCEPTION_IF_NULL(seed_param);
  MS_EXCEPTION_IF_NULL(offset_param);
  param_type seed;
  param_type offset;
  const auto param_size = sizeof(param_type);
  auto ret = memcpy_s(static_cast<void *>(&seed), param_size, static_cast<void *>(state), param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  ret = memcpy_s(static_cast<void *>(&offset), param_size, static_cast<void *>(state + param_size), param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  *seed_param = seed;
  *offset_param = offset;
  return PrepareOutput(seed, offset, outputs);
}

/*
  Input: cmd, seed_param, offset_param, new_seed
  Update seed, reset offset.
*/
bool ManualSeedCompute(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 4;
  constexpr size_t kSeedParamIdx = 1;
  constexpr size_t kOffsetParamIdx = 2;
  constexpr size_t kSeedIdx = 3;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "manual_seed");

  auto new_seed = GetDeviceAddress<param_type>(inputs, kSeedIdx);
  auto seed_param = GetDeviceAddress<param_type>(inputs, kSeedParamIdx);
  auto offset_param = GetDeviceAddress<param_type>(inputs, kOffsetParamIdx);
  MS_EXCEPTION_IF_NULL(new_seed);
  MS_EXCEPTION_IF_NULL(seed_param);
  MS_EXCEPTION_IF_NULL(offset_param);
  *seed_param = *new_seed;
  *offset_param = 0;
  return PrepareOutput(*new_seed, 0, outputs);
}

/*
  Input: cmd, seed_param, offset_param
  Return current seed
*/
bool InitialSeedCompute(const std::vector<kernel::KernelTensor *> &inputs,
                        const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 3;
  constexpr size_t kSeedParamIdx = 1;
  constexpr size_t kOffsetParamIdx = 2;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "initial_seed");

  auto seed = GetDeviceAddress<param_type>(inputs, kSeedParamIdx);
  auto offset = GetDeviceAddress<param_type>(inputs, kOffsetParamIdx);
  MS_EXCEPTION_IF_NULL(seed);
  MS_EXCEPTION_IF_NULL(offset);
  return PrepareOutput(*seed, *offset, outputs);
}
}  // namespace

bool GeneratorCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto cmd = GetDeviceAddress<int64_t>(inputs, kCmdIndex);
  MS_EXCEPTION_IF_NULL(cmd);
  static const std::unordered_map<int64_t, ComputeFunc> compute_map{{STEP, StepCompute},
                                                                    {SEED, SeedCompute},
                                                                    {GET_STATE, GetStateCompute},
                                                                    {SET_STATE, SetStateCompute},
                                                                    {MANUAL_SEED, ManualSeedCompute},
                                                                    {INITIAL_SEED, InitialSeedCompute}};
  auto iter = compute_map.find(*cmd);
  if (iter == compute_map.end()) {
    MS_LOG(ERROR) << "Unknown cmd: " << *cmd;
    return false;
  }
  return (iter->second)(inputs, outputs);
}

std::vector<KernelAttr> GeneratorCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> kernel_attrs{KernelAttr().AddSkipCheckAttr(true)};
  return kernel_attrs;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Generator, GeneratorCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
