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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/base/arithmetic_base.h"

namespace mindspore::kernel {
class ArithmeticCPUKernel : public ArithmeticBaseCPUKernel {
 public:
  ArithmeticCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ArithmeticBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ArithmeticCPUKernel() override = default;
  int ReSize() override;
  int Run() override;

 protected:
  void DoBroadcast(void *out_data, int input_index) override;
  int DoExecute(const void *input0, const void *input1, void *output, int64_t size) override;
  void InitRunFunction(int primitive_type) override;

 private:
  typedef struct {
    int primitive_type_;
    int activation_type_;
    ArithmeticFunc<float> func_;
    ArithmeticFunc<int> int_func_;
    ArithmeticFunc<bool> bool_func_;
    ArithmeticOptFunc<float> opt_func_;
    ArithmeticOptFunc<int> opt_int_func_;
    ArithmeticOptFunc<bool> opt_bool_func_;
  } ARITHMETIC_FUNC_INFO_FP32;

  ArithmeticFunc<float> arithmetic_run_fp32_{nullptr};
  ArithmeticOptFunc<float> arithmetic_opt_run_fp32_{nullptr};
  ArithmeticFunc<int> arithmetic_run_int_{nullptr};
  ArithmeticOptFunc<int> arithmetic_opt_run_int_{nullptr};
  ArithmeticFunc<bool> arithmetic_run_bool_{nullptr};
  ArithmeticOptFunc<bool> arithmetic_opt_run_bool_{nullptr};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ARITHMETIC_FP32_H_
