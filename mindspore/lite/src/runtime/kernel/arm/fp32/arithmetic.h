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
#include "src/runtime/kernel/arm/nnacl/fp32/arithmetic.h"
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
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_Sub;

namespace mindspore::kernel {
class ArithmeticCPUKernel : public LiteKernel {
  typedef int (*ArithmeticRun)(float *input0, float *input1, float *output, int element_size);
  typedef int (*ArithmeticOptRun)(float *input0, float *input1, float *output, int element_size,
                                  ArithmeticParameter *param);
  typedef int (*ArithmeticBroadcastRun)(float *input0, float *input1, float *tile_input0, float *tile_input1,
                                        float *output, int element_size, ArithmeticParameter *param);

 public:
  ArithmeticCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), thread_count_(ctx->thread_num_) {
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
    switch (parameter->type_) {
      case PrimitiveType_Mul:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_run_ = ElementMulRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmetic_run_ = ElementMulRelu6;
            break;
          default:
            arithmetic_run_ = ElementMul;
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
            break;
        }
        break;
      case PrimitiveType_Div:
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
        arithmetic_broadcast_run_ = BroadcastLogicalAnd;
        break;
      case PrimitiveType_LogicalOr:
        arithmetic_run_ = ElementLogicalOr;
        arithmetic_broadcast_run_ = BroadcastLogicalOr;
        break;
      case PrimitiveType_Maximum:
        arithmetic_run_ = ElementMaximum;
        arithmetic_broadcast_run_ = BroadcastMaximum;
        break;
      case PrimitiveType_Minimum:
        arithmetic_run_ = ElementMinimum;
        arithmetic_broadcast_run_ = BroadcastMinimum;
        break;
      case PrimitiveType_FloorDiv:
        arithmetic_run_ = ElementFloorDiv;
        arithmetic_broadcast_run_ = BroadcastFloorDiv;
        break;
      case PrimitiveType_FloorMod:
        arithmetic_run_ = ElementFloorMod;
        arithmetic_broadcast_run_ = BroadcastFloorMod;
        break;
      case PrimitiveType_Equal:
        arithmetic_run_ = ElementEqual;
        arithmetic_broadcast_run_ = BroadcastEqual;
        break;
      case PrimitiveType_NotEqual:
        arithmetic_run_ = ElementNotEqual;
        arithmetic_broadcast_run_ = BroadcastNotEqual;
        break;
      case PrimitiveType_Less:
        arithmetic_run_ = ElementLess;
        arithmetic_broadcast_run_ = BroadcastLess;
        break;
      case PrimitiveType_LessEqual:
        arithmetic_run_ = ElementLessEqual;
        arithmetic_broadcast_run_ = BroadcastLessEqual;
        break;
      case PrimitiveType_Greater:
        arithmetic_run_ = ElementGreater;
        arithmetic_broadcast_run_ = BroadcastGreater;
        break;
      case PrimitiveType_GreaterEqual:
        arithmetic_run_ = ElementGreaterEqual;
        arithmetic_broadcast_run_ = BroadcastGreaterEqual;
        break;
      case PrimitiveType_SquaredDifference:
        arithmetic_run_ = ElementSquaredDifference;
        arithmetic_broadcast_run_ = BroadcastSquaredDifference;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << parameter->type_;
        arithmetic_run_ = nullptr;
        arithmetic_broadcast_run_ = nullptr;
        break;
    }
  }
  ~ArithmeticCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoArithmetic(int task_id);

 private:
  void FreeTileData();
  int thread_count_;
  float *tile_data0_ = nullptr;
  float *tile_data1_ = nullptr;
  ArithmeticParameter *arithmeticParameter_;
  ArithmeticRun arithmetic_run_ = nullptr;
  ArithmeticBroadcastRun arithmetic_broadcast_run_ = nullptr;
  ArithmeticOptRun arithmetic_opt_run_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_H_
