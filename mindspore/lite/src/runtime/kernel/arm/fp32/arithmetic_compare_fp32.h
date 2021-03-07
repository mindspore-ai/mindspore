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
#include "nnacl/fp32/arithmetic_compare_fp32.h"

namespace mindspore::kernel {
typedef int (*ArithmeticCompareFp32Func)(const float *input0, const float *input1, uint8_t *output, int element_size);
typedef int (*ArithmeticCompareIntFunc)(const int *input0, const int *input1, uint8_t *output, int element_size);
class ArithmeticCompareCPUKernel : public ArithmeticCPUKernel {
 public:
  explicit ArithmeticCompareCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ArithmeticCPUKernel(parameter, inputs, outputs, ctx) {
    switch (parameter->type_) {
      case PrimitiveType_Equal:
        func_fp32_ = ElementEqualFp32;
        func_int32_ = ElementEqualInt32;
        break;
      case PrimitiveType_NotEqual:
        func_fp32_ = ElementNotEqualFp32;
        func_int32_ = ElementNotEqualInt32;
        break;
      case PrimitiveType_Less:
        func_fp32_ = ElementLessFp32;
        func_int32_ = ElementLessInt32;
        break;
      case PrimitiveType_LessEqual:
        func_fp32_ = ElementLessEqualFp32;
        func_int32_ = ElementLessEqualInt32;
        break;
      case PrimitiveType_Greater:
        func_fp32_ = ElementGreaterFp32;
        func_int32_ = ElementGreaterInt32;
        break;
      case PrimitiveType_GreaterEqual:
        func_fp32_ = ElementGreaterEqualFp32;
        func_int32_ = ElementGreaterEqualInt32;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << parameter->type_;
        func_fp32_ = nullptr;
        func_int32_ = nullptr;
        break;
    }
  }
  ~ArithmeticCompareCPUKernel() override = default;

  int DoArithmetic(int task_id) override;
  int BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count, int out_thread_stride) override;

 private:
  ArithmeticCompareFp32Func func_fp32_ = nullptr;
  ArithmeticCompareIntFunc func_int32_ = nullptr;
};
int ArithmeticCompareRun(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_COMPARE_H_
