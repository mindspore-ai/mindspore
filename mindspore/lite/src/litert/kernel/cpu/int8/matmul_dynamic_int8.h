/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_INT8_H_

#include <vector>
#include "src/litert/kernel/cpu/int8/matmul_dynamic_base_int8.h"

namespace mindspore::kernel {
class MatmulDynamicInt8CPUKernel : public MatmulDynamicBaseInt8CPUKernel {
 public:
  MatmulDynamicInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : MatmulDynamicBaseInt8CPUKernel(parameter, inputs, outputs, ctx) {}
  ~MatmulDynamicInt8CPUKernel() override = default;
  int Run() override;

 public:
  int RunImpl(int task_id);

 private:
  void InitParameter() override;

 private:
  PackFunc a_pack_func_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_INT8_H_
