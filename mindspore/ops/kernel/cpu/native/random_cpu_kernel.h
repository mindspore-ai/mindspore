/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_RANDOM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_RANDOM_CPU_KERNEL_H_
#include <utility>
#include <vector>
#include <string>
#include <random>
#include <map>
#include "kernel/cpu/cpu_kernel.h"
#include "primitive/random_ops.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace random_cpu {
constexpr auto kStandardNormal = "StandardNormal";
constexpr auto kUniformInt = "UniformInt";
constexpr auto kUniformReal = "UniformReal";
enum RandomOptype { RANDOM_OP_NORMAL = 0, RANDOM_OP_UNIFORM_INT, RANDOM_OP_UNIFORM_REAL, RANDOM_OP_INVALID_TYPE = 255 };

const std::map<std::string, RandomOptype> kRandomOpTypeMap = {
  {"StandardNormal", RANDOM_OP_NORMAL}, {"UniformInt", RANDOM_OP_UNIFORM_INT}, {"UniformReal", RANDOM_OP_UNIFORM_REAL}};

class RandomCpuKernelMod : public NativeCpuKernelMod {
 public:
  RandomCpuKernelMod() = default;
  explicit RandomCpuKernelMod(const std::string &kernel_name) : kernel_type_(kernel_name) {}
  ~RandomCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  RandomOptype random_op_type_{RANDOM_OP_INVALID_TYPE};
  std::string kernel_type_;
  std::mt19937 mtrng_;
  std::default_random_engine dfrng_;
};
}  // namespace random_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_RANDOM_CPU_KERNEL_H_
