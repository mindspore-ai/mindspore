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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_FP16_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_FP16_GRAD_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "schema/model_generated.h"

using mindspore::schema::PrimitiveType_AddGrad;
using mindspore::schema::PrimitiveType_DivGrad;
using mindspore::schema::PrimitiveType_MaximumGrad;
using mindspore::schema::PrimitiveType_MinimumGrad;
using mindspore::schema::PrimitiveType_MulGrad;
using mindspore::schema::PrimitiveType_SubGrad;

namespace mindspore::kernel {

class ArithmeticGradCPUKernelFp16;

class ArithmeticGradCPUKernelFp16 : public InnerKernel {
  typedef int (ArithmeticGradCPUKernelFp16::*ArithmeticGradOperation)(float16_t *, int, float16_t *, int, float16_t *,
                                                                      int);

 public:
  explicit ArithmeticGradCPUKernelFp16(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx), tile_data0(NULL), tile_data1(NULL), tile_data2(NULL) {
    switch (type()) {
      case PrimitiveType_MaximumGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernelFp16::ArithmeticGradMaximum;
        break;
      case PrimitiveType_MinimumGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernelFp16::ArithmeticGradMinimum;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << parameter->type_;
        break;
    }
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~ArithmeticGradCPUKernelFp16() override {
    if (tile_data0) delete[] tile_data0;
    if (tile_data1) delete[] tile_data1;
    if (tile_data2) delete[] tile_data2;
  }

  int Init() override;
  int InferShape();
  int ReSize() override;
  int Run() override;
  int Execute(int task_id);

 private:
  int ArithmeticGradMaximum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size, float16_t *dx2, int dx2_size);
  int ArithmeticGradMinimum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size, float16_t *dx2, int dx2_size);
  ArithmeticParameter *arithmeticParameter_;
  ArithmeticGradOperation arithmetic_grad_;
  float16_t *tile_data0;
  float16_t *tile_data1;
  float16_t *tile_data2;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_FP16_GRAD_H_
