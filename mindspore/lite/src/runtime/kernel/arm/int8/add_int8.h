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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ADD_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ADD_INT8_H_

#include <vector>
#include <limits>
#include <algorithm>
#include "src/lite_kernel.h"
#include "nnacl/int8/add_int8.h"
#include "nnacl/arithmetic.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
class QuantizedAddCPUKernel : public LiteKernel {
 public:
  explicit QuantizedAddCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    arith_para_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~QuantizedAddCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int tId);

 private:
  void BroadcastRun(int task_id);

 private:
  AddQuantParameter para_;
  ArithmeticParameter *arith_para_ = nullptr;
  int in_size_ = 0;
  int out_size_ = 0;
  int thread_count_ = 1;
  int elements_num_ = 0;
  bool support_opt_add_ = false;
  int8_t *input0_data_ = nullptr;
  int8_t *input1_data_ = nullptr;
  int8_t *output_data_ = nullptr;
};

int AddInt8Run(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ADD_INT8_H_
