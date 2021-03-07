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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SPLIT_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SPLIT_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/split_base.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class SplitInt8CPUKernel : public SplitBaseCPUKernel {
 public:
  SplitInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : SplitBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~SplitInt8CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int Split(int task_id);

 private:
  int8_t *input_ptr_;
  std::vector<int8_t *> output_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SPLIT_INT8_H_
