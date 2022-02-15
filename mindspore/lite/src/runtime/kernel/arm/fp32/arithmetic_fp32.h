/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_FP32_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/arithmetic_fp32.h"

using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_DivFusion;
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
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
class ArithmeticCPUKernel : public InnerKernel {
  typedef int (*ArithmeticRun)(const float *input0, const float *input1, float *output, const int element_size);
  typedef int (*ArithmeticOptRun)(const float *input0, const float *input1, float *output, const int element_size,
                                  const ArithmeticParameter *param);
  typedef int (*ArithmeticIntRun)(const int *input0, const int *input1, int *output, const int element_size);
  typedef int (*ArithmeticOptIntRun)(const int *input0, const int *input1, int *output, const int element_size,
                                     const ArithmeticParameter *param);
  typedef int (*ArithmeticBoolRun)(const bool *input0, const bool *input1, bool *output, const int element_size);
  typedef int (*ArithmeticOptBoolRun)(const bool *input0, const bool *input1, bool *output, const int element_size,
                                      const ArithmeticParameter *param);

  typedef struct {
    int primitive_type_;
    int activation_type_;
    ArithmeticRun func_;
    ArithmeticIntRun int_func_;
    ArithmeticBoolRun bool_func_;
    ArithmeticOptRun opt_func_;
    ArithmeticOptIntRun opt_int_func_;
    ArithmeticOptBoolRun opt_bool_func_;
  } ARITHMETIC_FUNC_INFO_FP32;

 public:
  ArithmeticCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~ArithmeticCPUKernel() { FreeConstTileBuff(); }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int DoArithmetic(int task_id);

 protected:
  virtual void InitRunFunction(int primitive_type);
  virtual int CheckDataType();
  virtual int ConstTensorBroadCast();
  virtual void TileConstTensor(const void *in_data, void *out_data, size_t ndim, const int *in_shape,
                               const int *in_strides, const int *out_strides, const int *multiple);
  virtual int DoExecute(const void *input0, const void *input1, void *output, int size, bool is_opt);
  virtual bool IsBatchScalarCalc();
  virtual bool IsScalarClac();
  virtual int CalcArithmeticByBatch(int task_id);
  bool input0_broadcast_ = false;
  bool input1_broadcast_ = false;
  void *input0_ptr_ = nullptr;
  void *input1_ptr_ = nullptr;
  void *output_ptr_ = nullptr;
  uint8_t *batch_a_ptr_ = nullptr;
  uint8_t *batch_b_ptr_ = nullptr;
  uint8_t *batch_c_ptr_ = nullptr;
  int break_pos_ = 0;
  ArithmeticParameter *param_ = nullptr;
  int data_type_len_ = sizeof(float);
  int out_batch_ = 1;
  int a_stride_size_ = 1;
  int b_stride_size_ = 1;
  int c_stride_size_ = 1;
  int last_batch_axis_ = 0;
  bool scalar_ = false;
  bool batch_scalar_ = false;
  bool split_by_batch_ = false;
  std::vector<int> a_offset_;
  std::vector<int> b_offset_;

 private:
  int InitIndexOffsetInfo();
  int BatchScalarCalc(int task_id);
  int BiasCalc(int task_id);
  void FreeConstTileBuff();
  bool IsBiasCalc() const;
  ArithmeticRun arithmetic_run_ = nullptr;
  ArithmeticOptRun arithmetic_opt_run_ = nullptr;
  ArithmeticIntRun arithmetic_run_int_ = nullptr;
  ArithmeticOptIntRun arithmetic_opt_run_int_ = nullptr;
  ArithmeticBoolRun arithmetic_run_bool_ = nullptr;
  ArithmeticOptBoolRun arithmetic_opt_run_bool_ = nullptr;
};
int ArithmeticsRun(void *cdata, int task_id, float lhs_scale, float rhs_scale);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_FP32_H_
