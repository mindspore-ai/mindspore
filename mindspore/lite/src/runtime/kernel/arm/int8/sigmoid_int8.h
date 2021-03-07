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

#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_SIGMOID_INT8_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_SIGMOID_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/int8/sigmoid_int8.h"

namespace mindspore::kernel {
class SigmoidInt8CPUKernel : public LiteKernel {
 public:
  SigmoidInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~SigmoidInt8CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  int8_t table_list_[256]{0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_SIGMOID_INT8_H_
