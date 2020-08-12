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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/nnacl/fp32/arithmetic_self.h"
#include "src/runtime/kernel/arm/nnacl/arithmetic_self_parameter.h"
#include "schema/model_generated.h"
#include "include/context.h"

using mindspore::lite::Context;
using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Exp;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {
class ArithmeticSelfCPUKernel : public LiteKernel {
  typedef int (*ArithmeticSelfRun)(float *input, float *output, int element_size);

 public:
  explicit ArithmeticSelfCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                                   const lite::Primitive *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), ctx_(ctx), thread_count_(ctx->thread_num_) {
    switch (parameter->type_) {
      case PrimitiveType_Abs:
        arithmeticSelf_run_ = ElementAbs;
        break;
      case PrimitiveType_Cos:
        arithmeticSelf_run_ = ElementCos;
        break;
      case PrimitiveType_Exp:
        arithmeticSelf_run_ = ElementExp;
        break;
      case PrimitiveType_Log:
        arithmeticSelf_run_ = ElementLog;
        break;
      case PrimitiveType_Square:
        arithmeticSelf_run_ = ElementSquare;
        break;
      case PrimitiveType_Sqrt:
        arithmeticSelf_run_ = ElementSqrt;
        break;
      case PrimitiveType_Rsqrt:
        arithmeticSelf_run_ = ElementRsqrt;
        break;
      case PrimitiveType_Sin:
        arithmeticSelf_run_ = ElementSin;
        break;
      case PrimitiveType_LogicalNot:
        arithmeticSelf_run_ = ElementLogicalNot;
        break;
      case PrimitiveType_Floor:
        arithmeticSelf_run_ = ElementFloor;
        break;
      case PrimitiveType_Ceil:
        arithmeticSelf_run_ = ElementCeil;
        break;
      case PrimitiveType_Round:
        arithmeticSelf_run_ = ElementRound;
        break;
      default:
        break;
    }
    arithmeticSelfParameter_ = reinterpret_cast<ArithmeticSelfParameter *>(parameter);
  }
  ~ArithmeticSelfCPUKernel() override = default;

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
  ArithmeticSelfRun arithmeticSelf_run_;
  const Context *ctx_;
  float *in_ptr_;
  float *out_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_
