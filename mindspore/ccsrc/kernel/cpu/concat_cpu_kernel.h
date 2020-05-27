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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_CONCAT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_CONCAT_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ConcatCPUKernel : public CPUKernel {
 public:
  ConcatCPUKernel() : axis_(0) {}
  ~ConcatCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  void CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1, size_t dim2,
                        float **output_addr, size_t *buff_size);
  int axis_;
  std::vector<std::vector<size_t>> input_shape_list_;
  std::vector<size_t> output_shape_;
};

MS_REG_CPU_KERNEL(Concat,
                  KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ConcatCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_CONCAT_CPU_KERNEL_H_
