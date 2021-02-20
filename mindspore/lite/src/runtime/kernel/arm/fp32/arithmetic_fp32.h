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
using mindspore::schema::PrimitiveType_Mod;
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
  typedef int (*ArithmeticBoolRun)(const bool *input0, const bool *input1, bool *output, const int element_size);

  typedef struct {
    int primitive_type_;
    int activation_type_;
    ArithmeticRun func_;
    ArithmeticIntRun int_func_;
    ArithmeticBoolRun bool_func_;
    ArithmeticOptRun opt_func_;
    ArithmeticOptIntRun opt_int_func_;
  } ARITHMETIC_FUNC_INFO_FP32;

 public:
  ArithmeticCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), thread_count_(ctx->thread_num_) {
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
    InitRunFunction();
  }
  ~ArithmeticCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  virtual int DoArithmetic(int task_id);
  virtual int BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count, int out_thread_stride);

 private:
  void InitRunFunction();
  void InitParam();
  void FreeTmpPtr();
  int CheckDataType();
  int InitBroadCastCase();
  void InitParamInRunTime();
  bool CanBatchScalar();
  int BatchScalarCalc(int task_id);

 protected:
  bool input0_broadcast_ = false;
  bool input1_broadcast_ = false;
  void *input0_ptr_ = nullptr;
  void *input1_ptr_ = nullptr;
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
  ArithmeticBoolRun arithmetic_run_bool_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_H_
