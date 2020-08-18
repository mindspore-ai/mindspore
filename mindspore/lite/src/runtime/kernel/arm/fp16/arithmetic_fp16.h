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
#include "src/runtime/kernel/arm/nnacl/fp16/arithmetic_fp16.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
class ArithmeticFP16CPUKernel : public LiteKernel {
  typedef int (*ArithmeticRun)(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
  typedef int (*ArithmeticOptRun)(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                                  ArithmeticParameter *param);

 public:
  ArithmeticFP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
        arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
      }
  ~ArithmeticFP16CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoArithmetic(int task_id);

 private:
  void FreeTmpBuffer();
  float16_t *tile_data0_ = nullptr;
  float16_t *tile_data1_ = nullptr;
  float16_t *input0_fp16_ = nullptr;
  float16_t *input1_fp16_ = nullptr;
  float16_t *output_fp16_ = nullptr;
  ArithmeticParameter *arithmeticParameter_ = nullptr;
  ArithmeticRun arithmetic_run_ = nullptr;
  ArithmeticOptRun arithmetic_opt_run_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_
