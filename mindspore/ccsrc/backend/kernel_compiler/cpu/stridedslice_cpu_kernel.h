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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "nnacl/fp32/strided_slice_fp32.h"

namespace mindspore {
namespace kernel {
class StridedSliceCPUKernel : public CPUKernel {
 public:
  StridedSliceCPUKernel() = default;
  ~StridedSliceCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  enum ParallelStrategy { kOnSplitAxis, kOnOuter };
  void InitSliceParam(const std::vector<int64_t> &begin, const std::vector<int64_t> &end,
                      const std::vector<int64_t> &stride);
  bool MatchParallelPattern();
  void InitParallelParam();
  void ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num);
  int RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);
  int RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);

  TypeId dtype_;
  int data_size_{4};
  int split_axis_{-1};
  int inner_{1};
  int outer_{1};
  int cal_num_per_thread_{1};
  bool parallel_{false};
  ParallelStrategy parallel_strategy_{kOnSplitAxis};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  StridedSliceParameter slice_param_;
};

MS_REG_CPU_KERNEL(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  StridedSliceCPUKernel);
MS_REG_CPU_KERNEL(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  StridedSliceCPUKernel);
MS_REG_CPU_KERNEL(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  StridedSliceCPUKernel);
MS_REG_CPU_KERNEL(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  StridedSliceCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_
