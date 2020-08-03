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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARITHMETIC_SELF_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARITHMETIC_SELF_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/opclib/arithmetic_self_parameter.h"
#include "src/runtime/kernel/arm/opclib/int8/arithmetic_self_int8.h"
#include "schema/model_generated.h"
#include "include/context.h"


using mindspore::lite::Context;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Ceil;

namespace mindspore::kernel {
class ArithmeticSelfInt8CPUKernel : public LiteKernel {
  typedef int (*ArithmeticSelfInt8Run)(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para);

 public:
  explicit ArithmeticSelfInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx)
    : LiteKernel(parameter, inputs, outputs), ctx_(ctx), thread_count_(ctx->threadNum) {
    switch (parameter->type_) {
      case PrimitiveType_Round:
        arithmeticSelf_run_ = ElementRound;
        break;
      case PrimitiveType_Floor:
        arithmeticSelf_run_ = ElementFloor;
        break;
      case PrimitiveType_Ceil:
        arithmeticSelf_run_ = ElementCeil;
        break;
      default:
        break;
    }
    arithmeticSelfParameter_ = reinterpret_cast<ArithmeticSelfParameter *>(parameter);
  }
  ~ArithmeticSelfInt8CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoArithmeticSelf(int task_id);

 private:
  int thread_count_;
  int thread_sz_count_;
  int thread_sz_stride_;
  size_t data_size_;
  ArithmeticSelfParameter *arithmeticSelfParameter_;
  ArithmeticSelfInt8Run arithmeticSelf_run_;
  const Context *ctx_;
  int8_t *in_ptr_;
  int8_t *out_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARITHMETIC_SELF_INT8_H_

