/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRANSPOSE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRANSPOSE_FP32_H_

#ifndef BFC_MEMORY
#include <vector>
#include "src/litert/kernel/cpu/base/transpose_base.h"

namespace mindspore::kernel {
class TransposeCPUKernel : public TransposeBaseCPUKernel {
 public:
  explicit TransposeCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TransposeBaseCPUKernel(param, inputs, outputs, ctx) {}
  ~TransposeCPUKernel() override = default;

  int ReSize() override;
  int DoTransposeMultiThread(int task_id) override;

 private:
  int DoTransposeSingleThread() override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRANSPOSE_FP32_H_
#endif
