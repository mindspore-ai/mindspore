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

/*
  Input: cmd, seed_param, offset_param, step_size
  Output: seed, offset
  Update offset parameter and return the old offset value.
*/
bool StepCompute(const std::vector<kernel::KernelTensor *> &inputs,
                 const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 4;
  constexpr size_t kOutputNum = 2;
  constexpr size_t kSeedParamIdx = 1;
  constexpr size_t kOffsetParamIdx = 2;
  constexpr size_t kStepSizeIdx = 3;
  constexpr size_t kOutputSeedIdx = 0;
  constexpr size_t kOutputOffsetIdx = 1;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "step");

  auto seed_param = reinterpret_cast<param_type *>(inputs[kSeedParamIdx]->device_ptr());
  auto offset_param = reinterpret_cast<param_type *>(inputs[kOffsetParamIdx]->device_ptr());
  auto step_size = *reinterpret_cast<param_type *>(inputs[kStepSizeIdx]->device_ptr());
  auto seed_output = reinterpret_cast<param_type *>(outputs[kOutputSeedIdx]->device_ptr());
  auto offset_output = reinterpret_cast<param_type *>(outputs[kOutputOffsetIdx]->device_ptr());
  *seed_output = *seed_param;
  auto old_offset = *offset_param;
  *offset_output = old_offset;
  *offset_param = old_offset + step_size;
  return true;
}

/*
  Input: cmd, seed, offset
  Output: seed
  Generate random number as new seed. Reset offset.
*/
bool SeedCompute(const std::vector<kernel::KernelTensor *> &inputs,
                 const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 3;
  constexpr size_t kOutputNum = 1;
  constexpr size_t kSeedIdx = 1;
  constexpr size_t kOffsetIdx = 2;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "seed");

  auto seed = reinterpret_cast<param_type *>(inputs[kSeedIdx]->device_ptr());
  auto offset = reinterpret_cast<param_type *>(inputs[kOffsetIdx]->device_ptr());
  auto output = reinterpret_cast<param_type *>(outputs.front()->device_ptr());
  auto rd = static_cast<param_type>(std::random_device()());
  *seed = rd;
  *output = rd;
  *offset = 0;
  return true;
}

/*
  Input: cmd, seed, offset
  Output: state tensor
  Return a tensor representing the generator state.
*/
bool GetStateCompute(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 3;
  constexpr size_t kOutputNum = 1;
  constexpr size_t kSeedIdx = 1;
  constexpr size_t kOffsetIdx = 2;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "get_state");

  auto seed = reinterpret_cast<void *>(inputs[kSeedIdx]->device_ptr());
  auto offset = reinterpret_cast<void *>(inputs[kOffsetIdx]->device_ptr());
  auto output = reinterpret_cast<uint8_t *>(outputs.front()->device_ptr());
  auto output_size = outputs.front()->GetShapeVector().front() * sizeof(state_type);
  auto param_size = sizeof(param_type);
  auto ret = memcpy_s(static_cast<void *>(output), output_size, seed, param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  ret = memcpy_s(static_cast<void *>(output + param_size), output_size - param_size, offset, param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  return true;
}

/*
  Input: cmd, state
  Output: seed, offset
  Unpack state to seed and offset
*/
bool UnpackStateCompute(const std::vector<kernel::KernelTensor *> &inputs,
                        const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 2;
  constexpr size_t kOutputNum = 2;
  constexpr size_t kStateIdx = 1;
  constexpr size_t kSeedIdx = 0;
  constexpr size_t kOffsetIdx = 1;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "unpack_state");

  auto state = reinterpret_cast<state_type *>(inputs[kStateIdx]->device_ptr());
  auto seed = reinterpret_cast<void *>(outputs[kSeedIdx]->device_ptr());
  auto offset = reinterpret_cast<void *>(outputs[kOffsetIdx]->device_ptr());
  auto param_size = sizeof(param_type);
  auto ret = memcpy_s(seed, param_size, static_cast<void *>(state), param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  ret = memcpy_s(offset, param_size, static_cast<void *>(state + param_size), param_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kGenerator << ", memcpy failed.";
  }
  return true;
}

/*
  Input: cmd, seed_param, offset_param, seed, offset,
  Output: None
  Update seed and offset parameters with seed and offset tensors.
*/
bool SetStateCompute(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 5;
  constexpr size_t kOutputNum = 1;
  constexpr size_t kSeedParamIdx = 1;
  constexpr size_t kOffsetParamIdx = 2;
  constexpr size_t kSeedIdx = 3;
  constexpr size_t kOffsetIdx = 4;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "set_state");

  auto seed = reinterpret_cast<param_type *>(inputs[kSeedIdx]->device_ptr());
  auto offset = reinterpret_cast<param_type *>(inputs[kOffsetIdx]->device_ptr());
  auto seed_param = reinterpret_cast<param_type *>(inputs[kSeedParamIdx]->device_ptr());
  auto offset_param = reinterpret_cast<param_type *>(inputs[kOffsetParamIdx]->device_ptr());
  *seed_param = *seed;
  *offset_param = *offset;
  return true;
}

/*
  Input: cmd, seed_param
  Output: None
  Return current seed
*/
bool InitialSeedCompute(const std::vector<kernel::KernelTensor *> &inputs,
                        const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 2;
  constexpr size_t kOutputNum = 1;
  constexpr size_t kSeedParamIdx = 1;
  GeneratorCheck(inputs, outputs, kInputNum, kOutputNum, "initial_seed");

  auto seed_param = reinterpret_cast<param_type *>(inputs[kSeedParamIdx]->device_ptr());
  auto output = reinterpret_cast<param_type *>(outputs.front()->device_ptr());
  *output = *seed_param;
  return true;
}
}  // namespace

bool GeneratorCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto cmd = *reinterpret_cast<int64_t *>(inputs[kCmdIndex]->device_ptr());
  static const std::unordered_map<int64_t, ComputeFunc> compute_map{{STEP, StepCompute},
                                                                    {SEED, SeedCompute},
                                                                    {GET_STATE, GetStateCompute},
                                                                    {SET_STATE, SetStateCompute},
                                                                    {UNPACK_STATE, UnpackStateCompute},
                                                                    {INITIAL_SEED, InitialSeedCompute}};
  auto iter = compute_map.find(cmd);
  if (iter == compute_map.end()) {
    MS_LOG(ERROR) << "Unknown cmd: " << cmd;
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
