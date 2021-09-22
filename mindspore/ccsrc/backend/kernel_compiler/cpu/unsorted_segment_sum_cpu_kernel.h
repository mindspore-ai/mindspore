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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNSORTED_SEGMENT_SUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNSORTED_SEGMENT_SUM_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "nnacl/base/unsorted_segment_sum_base.h"

namespace mindspore {
namespace kernel {
class UnsortedSegmentSumCPUKernel : public CPUKernel {
 public:
  UnsortedSegmentSumCPUKernel() = default;
  ~UnsortedSegmentSumCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId dtype_{kTypeUnknown};
  TypeId segment_ids_dtype_{kTypeUnknown};
  size_t unit_num_{1};
  size_t input_dim1_{1};
  size_t output_dim0_{1};
  size_t output_dim1_{1};
};

MS_REG_CPU_KERNEL(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentSumCPUKernel);
MS_REG_CPU_KERNEL(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentSumCPUKernel);
MS_REG_CPU_KERNEL(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentSumCPUKernel);
MS_REG_CPU_KERNEL(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentSumCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNSORTED_SEGMENT_SUM_CPU_KERNEL_H_
