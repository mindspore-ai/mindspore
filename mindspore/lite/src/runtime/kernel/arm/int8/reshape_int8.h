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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESHAPE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESHAPE_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "nnacl/reshape_parameter.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class ReshapeInt8CPUKernel : public LiteKernel {
 public:
  ReshapeInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    reshape_param_ = reinterpret_cast<ReshapeParameter *>(op_parameter_);
  }
  ~ReshapeInt8CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int64_t elements_num_;
  int64_t count_unit_;
  int8_t *input_data_ = nullptr;
  int8_t *output_data_ = nullptr;
  ReshapeParameter *reshape_param_ = nullptr;
};

int ReshapeInt8Run(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESHAPE_INT8_H_
