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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/arithmetic_fp32.h"
#include "nnacl/fp16/arithmetic_fp16.h"

namespace mindspore::kernel {
typedef int (*ArithmeticFuncFp16)(const float16_t *input0, const float16_t *input1, float16_t *output,
                                  int element_size);
typedef int (*ArithmeticOptFuncFp16)(const float16_t *input0, const float16_t *input1, float16_t *output,
                                     int element_size, ArithmeticParameter *param);
typedef struct {
  int primitive_type_;
  int activation_type_;
  ArithmeticFuncFp16 func_;
  ArithmeticOptFuncFp16 opt_func_;
} ARITHMETIC_FUNC_INFO_FP16;

class ArithmeticFP16CPUKernel : public ArithmeticCPUKernel {
 public:
  ArithmeticFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ArithmeticCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ArithmeticFP16CPUKernel() = default;
  int ReSize() override;
  int Run() override;
  bool IsBatchScalarCalc() override;
  bool IsScalarClac() override;

 private:
  void InitRunFunction(int primitive_type) override;
  int CheckDataType() override;
  int ConstTensorBroadCast() override;
  void TileConstTensor(const void *in_data, void *out_data, size_t ndim, const int *in_shape, const int *in_strides,
                       const int *out_strides, const int *multiple) override;
  int Execute(const void *input0, const void *input1, void *output, int size, bool is_opt) override;
  void FreeFp16Buffer();
  ArithmeticFuncFp16 arithmetic_func_ = nullptr;
  ArithmeticOptFuncFp16 arithmetic_opt_func_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ARITHMETIC_FP16_H_
