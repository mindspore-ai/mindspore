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
#include "nnacl/arithmetic_self_parameter.h"
#include "nnacl/int8/arithmetic_self_int8.h"
#include "schema/model_generated.h"
#include "include/context.h"

using mindspore::lite::InnerContext;
using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {
class ArithmeticSelfInt8CPUKernel : public LiteKernel {
  typedef int (*ArithmeticSelfInt8Run)(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para);

 public:
  explicit ArithmeticSelfInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    switch (parameter->type_) {
      case PrimitiveType_Round:
        arithmeticSelf_run_ = Int8ElementRound;
        break;
      case PrimitiveType_Floor:
        arithmeticSelf_run_ = Int8ElementFloor;
        break;
      case PrimitiveType_Ceil:
        arithmeticSelf_run_ = Int8ElementCeil;
        break;
      case PrimitiveType_Abs:
        arithmeticSelf_run_ = Int8ElementAbs;
        break;
      case PrimitiveType_Sin:
        arithmeticSelf_run_ = Int8ElementSin;
        break;
      case PrimitiveType_Cos:
        arithmeticSelf_run_ = Int8ElementCos;
        break;
      case PrimitiveType_Log:
        arithmeticSelf_run_ = Int8ElementLog;
        break;
      case PrimitiveType_Sqrt:
        arithmeticSelf_run_ = Int8ElementSqrt;
        break;
      case PrimitiveType_Rsqrt:
        arithmeticSelf_run_ = Int8ElementRsqrt;
        break;
      case PrimitiveType_Square:
        arithmeticSelf_run_ = Int8ElementSquare;
        break;
      case PrimitiveType_LogicalNot:
        arithmeticSelf_run_ = Int8ElementLogicalNot;
        break;
      case PrimitiveType_Reciprocal:
        arithmeticSelf_run_ = Int8ElementReciprocal;
      default:
        break;
    }
    para_ = reinterpret_cast<ArithmeticSelfParameter *>(parameter);
  }
  ~ArithmeticSelfInt8CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoArithmeticSelf(int task_id);

 private:
  int thread_sz_count_;
  int thread_sz_stride_;
  size_t data_size_;
  ArithmeticSelfParameter *para_;
  ArithmeticSelfInt8Run arithmeticSelf_run_;
  int thread_count_;
  int8_t *in_ptr_;
  int8_t *out_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARITHMETIC_SELF_INT8_H_
