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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_COMPARE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_COMPARE_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/arithmetic_fp32.h"

namespace mindspore::kernel {
class ArithmeticCompareCPUKernel : public ArithmeticCPUKernel {
  typedef int (*ArithmeticCompareFp32Func)(const float *input0, const float *input1, uint8_t *output, int element_size);
  typedef int (*ArithmeticCompareIntFunc)(const int *input0, const int *input1, uint8_t *output, int element_size);
  typedef int (*ArithmeticOptCompareFp32Func)(const float *input0, const float *input1, uint8_t *output,
                                              int element_size, const ArithmeticParameter *param);
  typedef int (*ArithmeticOptCompareIntFunc)(const int *input0, const int *input1, uint8_t *output, int element_size,
                                             const ArithmeticParameter *param);
  typedef struct {
    int primitive_type_;
    ArithmeticCompareFp32Func func_;
    ArithmeticCompareIntFunc int_func_;
    ArithmeticOptCompareFp32Func opt_func_;
    ArithmeticOptCompareIntFunc opt_int_func_;
  } ARITHMETIC_COMEPARE_FUNC_INFO_FP32;

 public:
  explicit ArithmeticCompareCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ArithmeticCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ArithmeticCompareCPUKernel() override = default;

 protected:
  void InitRunFunction(int primitive_type) override;
  int DoExecute(const void *input0, const void *input1, void *output, int64_t size) override;

 private:
  ArithmeticCompareFp32Func func_fp32_{nullptr};
  ArithmeticCompareIntFunc func_int32_{nullptr};
  ArithmeticOptCompareFp32Func opt_func_fp32_{nullptr};
  ArithmeticOptCompareIntFunc opt_func_int32_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_COMPARE_FP32_H_
