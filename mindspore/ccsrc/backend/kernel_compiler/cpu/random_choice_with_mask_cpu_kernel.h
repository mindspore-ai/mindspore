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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
#include <vector>
#include <random>
#include <algorithm>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_INPUT_DIMS = 5;
constexpr size_t INPUT_NUM = 1;
constexpr size_t OUTPUT_NUM = 2;
constexpr size_t MAX_DIMENSION = 5;

class RandomChoiceWithMaskCPUKernel : public CPUKernel {
 public:
  RandomChoiceWithMaskCPUKernel() = default;
  ~RandomChoiceWithMaskCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void InitInputOutputSize(const CNodePtr &kernel_node) override;

 private:
  int32_t input_dim_size = 0;
  int32_t non_zero_num = 0;
  int32_t input_total_count = 1;
  bool padding_flag = false;
  int32_t output_length = 0;
  int32_t output_non_zero_length = 0;
  int32_t count_{0};
  std::vector<int32_t> dims_;
  size_t input_shape_size_{0};
  size_t seed_{0};
  size_t seed2_{0};
  int input_size_{1};
  std::mt19937 generator_;
};

MS_REG_CPU_KERNEL(
  RandomChoiceWithMask,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  RandomChoiceWithMaskCPUKernel);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
