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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_REDUCE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_REDUCE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/litert/kernel/cpu/fp32/reduce_fp32.h"

namespace mindspore::kernel {
class ReduceFp16CPUKernel : public ReduceCPUKernel {
  typedef int (*Fp16Reducer)(const int outer_size, const int inner_size, const int axis_size, const float16_t *src_data,
                             float16_t *dst_data, const int tid, const int thread_num);

 public:
  ReduceFp16CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ReduceCPUKernel(param, inputs, outputs, ctx) {}
  ~ReduceFp16CPUKernel() = default;

  int CallReduceUnit(int task_id) override;

 private:
  void InitialKernelList() override;
  void HandleASumAndSumSquare() override;
  int CalculateCoeffOutput() override;

  Fp16Reducer fp16_reducer_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_REDUCE_FP16_H_
