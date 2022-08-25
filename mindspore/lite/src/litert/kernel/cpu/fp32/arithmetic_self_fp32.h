/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_SELF_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_SELF_FP32_H_

#include <vector>
#include <map>
#include <memory>
#include "src/litert/lite_kernel.h"

using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Erf;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_IsFinite;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_Log1p;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {
typedef int (*ArithmeticSelfFunc)(const float *input, float *output, const int element_size);
typedef int (*ArithmeticSelfBoolFunc)(const bool *input, bool *output, const int element_size);
typedef int (*ArithmeticSelfIntFunc)(const int *input, int *output, const int element_size);
typedef int (*ArithmeticSelfFloatBoolFunc)(const float *input, bool *output, const int element_size);
class ArithmeticSelfCPUKernel : public LiteKernel {
 public:
  explicit ArithmeticSelfCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    func_ = GetArithmeticSelfFun(parameter->type_);
    func_bool_ = GetArithmeticSelfBoolFun(parameter->type_);
    func_int_ = GetArithmeticSelfIntFun(parameter->type_);
    func_float_bool_ = GetArithmeticSelfFloatBoolFun(parameter->type_);
  }
  ~ArithmeticSelfCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int DoExecute(int task_id);

 private:
  ArithmeticSelfFunc GetArithmeticSelfFun(int primitive_type) const;
  ArithmeticSelfBoolFunc GetArithmeticSelfBoolFun(int primitive_type) const;
  ArithmeticSelfIntFunc GetArithmeticSelfIntFun(int primitive_type) const;
  ArithmeticSelfFloatBoolFunc GetArithmeticSelfFloatBoolFun(int primitive_type) const;
  ArithmeticSelfFunc func_;
  ArithmeticSelfBoolFunc func_bool_;
  ArithmeticSelfIntFunc func_int_;
  ArithmeticSelfFloatBoolFunc func_float_bool_;
};
int ArithmeticSelfRun(void *cdata, int task_id, float lhs_scale, float rhs_scale);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_SELF_FP32_H_
