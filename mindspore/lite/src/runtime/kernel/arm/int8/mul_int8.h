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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MUL_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MUL_INT8_H_

#include <vector>
#include <limits>
#include <algorithm>
#include "src/lite_kernel.h"
#include "nnacl/mul_parameter.h"
#include "nnacl/int8/mul_int8.h"
#include "nnacl/int8/arithmetic_int8.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
class MulInt8CPUKernel : public LiteKernel {
 public:
  explicit MulInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx_->thread_num_) {
    tile_para = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~MulInt8CPUKernel() override{};

  int Init() override;
  int ReSize() override;
  void CheckSameShapeSize(std::vector<int> in_tensor0_shape, std::vector<int> in_tensor1_shape);
  void CheckIfFastImpl();
  int Run() override;
  int DoExecute(int task_id);
  int FastDoExecute(int task_id);

 private:
  const lite::InnerContext *ctx_ = nullptr;
  ArithmeticParameter *tile_para = nullptr;
  MulParameter para_;
  bool fast_hw_broadcast_ = false;
  bool input1_hw_broadcast_ = false;
  int thread_count_ = 1;
  int64_t elements_num_ = 0;
  int64_t count_unit_ = 0;
  int8_t *input0_data_ = nullptr;
  int8_t *input1_data_ = nullptr;
  int8_t *output_data_ = nullptr;
};

int MulInt8Run(void *cdata, int task_id);
int FastHWBroadcatMulInt8Run(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MUL_INT8_H_
