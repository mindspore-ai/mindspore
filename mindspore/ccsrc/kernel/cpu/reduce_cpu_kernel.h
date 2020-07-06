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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_REDUCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_REDUCE_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ReduceCPUKernel : public CPUKernel {
 public:
  ReduceCPUKernel() = default;
  ~ReduceCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void Transpose(const int size, const float *input, const std::vector<size_t> &input_shape,
                 const std::vector<size_t> &input_axis, const int shape_size, float *output);
  size_t reduce_type_;
  std::vector<size_t> axis_;
  std::vector<size_t> shape_;
  size_t left_dims_ = 1;
  size_t stride_ = 1;
};
MS_REG_CPU_KERNEL(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReduceCPUKernel);
MS_REG_CPU_KERNEL(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReduceCPUKernel);
MS_REG_CPU_KERNEL(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReduceCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_CPU_REDUCE_CPU_KERNEL_H_
