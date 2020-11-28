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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "schema/model_generated.h"

using mindspore::schema::PrimitiveType_Add;
using mindspore::schema::PrimitiveType_Div;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_Mul;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_Sub;

namespace mindspore::kernel {
class ArithmeticCPUKernel : public LiteKernel {
  typedef int (*ArithmeticRun)(const float *input0, const float *input1, float *output, const int element_size);
  typedef int (*ArithmeticOptRun)(const float *input0, const float *input1, float *output, const int element_size,
                                  const ArithmeticParameter *param);
  typedef int (*ArithmeticIntRun)(const int *input0, const int *input1, int *output, const int element_size);
  typedef int (*ArithmeticOptIntRun)(const int *input0, const int *input1, int *output, const int element_size,
                                     const ArithmeticParameter *param);

 public:
  ArithmeticCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), thread_count_(ctx->thread_num_) {
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
    switch (parameter->type_) {
      case PrimitiveType_Mul:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_run_ = ElementMulRelu;
            arithmetic_run_int_ = ElementMulReluInt;
            break;
          case schema::ActivationType_RELU6:
            arithmetic_run_ = ElementMulRelu6;
            arithmetic_run_int_ = ElementMulRelu6Int;
            break;
          default:
            arithmetic_run_ = ElementMul;
            arithmetic_run_int_ = ElementMulInt;
            break;
        }
        break;
      case PrimitiveType_Add:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_run_ = ElementAddRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmetic_run_ = ElementAddRelu6;
            break;
          default:
            arithmetic_run_ = ElementAdd;
            arithmetic_run_int_ = ElementAddInt;
            break;
        }
        break;
      case PrimitiveType_Sub:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_run_ = ElementSubRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmetic_run_ = ElementSubRelu6;
            break;
          default:
            arithmetic_run_ = ElementSub;
            arithmetic_run_int_ = ElementSubInt;
            break;
        }
        break;
      case PrimitiveType_Div:
      case PrimitiveType_RealDiv:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_run_ = ElementDivRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmetic_run_ = ElementDivRelu6;
            break;
          default:
            arithmetic_run_ = ElementDiv;
            break;
        }
        break;
      case PrimitiveType_LogicalAnd:
        arithmetic_run_ = ElementLogicalAnd;
        break;
      case PrimitiveType_LogicalOr:
        arithmetic_run_ = ElementLogicalOr;
        break;
      case PrimitiveType_Maximum:
        arithmetic_run_ = ElementMaximum;
        break;
      case PrimitiveType_Minimum:
        arithmetic_run_ = ElementMinimum;
        break;
      case PrimitiveType_FloorDiv:
        arithmetic_run_ = ElementFloorDiv;
        arithmetic_run_int_ = ElementFloorDivInt;
        break;
      case PrimitiveType_FloorMod:
        arithmetic_run_ = ElementFloorMod;
        arithmetic_run_int_ = ElementFloorModInt;
        break;
      case PrimitiveType_Equal:
        arithmetic_run_ = ElementEqual;
        break;
      case PrimitiveType_NotEqual:
        arithmetic_run_ = ElementNotEqual;
        break;
      case PrimitiveType_Less:
        arithmetic_run_ = ElementLess;
        break;
      case PrimitiveType_LessEqual:
        arithmetic_run_ = ElementLessEqual;
        break;
      case PrimitiveType_Greater:
        arithmetic_run_ = ElementGreater;
        break;
      case PrimitiveType_GreaterEqual:
        arithmetic_run_ = ElementGreaterEqual;
        break;
      case PrimitiveType_SquaredDifference:
        arithmetic_run_ = ElementSquaredDifference;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << parameter->type_;
        arithmetic_run_ = nullptr;
        break;
    }
  }
  ~ArithmeticCPUKernel() override;

  int Init() override;
  int PreProcess() override;
  int ReSize() override;
  int Run() override;
  virtual int DoArithmetic(int task_id);
  virtual int BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count, int out_thread_stride);

 protected:
  int break_pos_ = 0;
  int outside_ = 0;
  int thread_count_ = 1;
  ArithmeticParameter *arithmeticParameter_ = nullptr;
  LiteDataType data_type_ = kDataTypeFloat;

 private:
  ArithmeticRun arithmetic_run_ = nullptr;
  ArithmeticOptRun arithmetic_opt_run_ = nullptr;
  ArithmeticIntRun arithmetic_run_int_ = nullptr;
  ArithmeticOptIntRun arithmetic_opt_run_int_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_H_
