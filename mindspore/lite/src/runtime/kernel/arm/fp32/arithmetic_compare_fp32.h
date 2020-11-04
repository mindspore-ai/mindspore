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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_COMPARE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_COMPARE_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/arithmetic_fp32.h"

namespace mindspore::kernel {
typedef int (*ArithmeticCompareFp32Func)(const float *input0, const float *input1, uint8_t *output, int element_size);
class ArithmeticCompareCPUKernel : public ArithmeticCPUKernel {
 public:
  explicit ArithmeticCompareCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                      const mindspore::lite::PrimitiveC *primitive)
      : ArithmeticCPUKernel(parameter, inputs, outputs, ctx, primitive) {
    func_ = GetArithmeticCompareFun(parameter->type_);
  }
  ~ArithmeticCompareCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  virtual int DoExecute(int task_id);

 private:
  ArithmeticCompareFp32Func GetArithmeticCompareFun(int primitive_type);
  ArithmeticCompareFp32Func func_;
};
int ArithmeticCompareRun(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_COMPARE_H_
