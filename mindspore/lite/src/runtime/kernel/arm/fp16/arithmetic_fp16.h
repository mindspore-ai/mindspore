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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
typedef int (*ArithmeticFuncFp16)(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
typedef int (*ArithmeticOptFuncFp16)(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                                     ArithmeticParameter *param);
typedef struct {
  int primitive_type_;
  int activation_type_;
  ArithmeticFuncFp16 func_;
  ArithmeticOptFuncFp16 opt_func_;
} ARITHMETIC_FUNC_INFO_FP16;

class ArithmeticFP16CPUKernel : public LiteKernel {
 public:
  ArithmeticFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                          const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    param_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~ArithmeticFP16CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int CheckDataType();
  int DoArithmetic(int task_id);
  int BroadcastRun(float16_t *input0, float16_t *input1, float16_t *output, int dim, int out_count,
                   int out_thread_stride);

 private:
  void InitParam();
  void FreeTmpBuffer();
  int outside_;
  int break_pos_;
  bool is_input0_fp32_ = false;
  bool is_input1_fp32_ = false;
  bool is_output_fp32_ = false;
  float16_t *input0_fp16_ = nullptr;
  float16_t *input1_fp16_ = nullptr;
  float16_t *output_fp16_ = nullptr;
  ArithmeticParameter *param_ = nullptr;
  ArithmeticFuncFp16 arithmetic_func_ = nullptr;
  ArithmeticOptFuncFp16 arithmetic_opt_func_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_
