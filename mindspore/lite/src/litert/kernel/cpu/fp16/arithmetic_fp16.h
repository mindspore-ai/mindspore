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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_ARITHMETIC_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_ARITHMETIC_FP16_H_

#include <vector>
#include "src/litert/kernel/cpu/base/arithmetic_base.h"

namespace mindspore::kernel {
class ArithmeticFP16CPUKernel : public ArithmeticBaseCPUKernel {
 public:
  ArithmeticFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ArithmeticBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ArithmeticFP16CPUKernel() override = default;
  int ReSize() override;
  int Run() override;

 private:
  typedef struct {
    int primitive_type_;
    int activation_type_;
    ArithmeticFunc<float16_t> func_;
    ArithmeticOptFunc<float16_t> opt_func_;
  } ARITHMETIC_FUNC_INFO_FP16;

  void DoBroadcast(void *out_data, int input_index) override;
  int DoExecute(const void *input0, const void *input1, void *output, int64_t size) override;
  void InitRunFunction(int primitive_type) override;
  ArithmeticFunc<float16_t> arithmetic_run_fp16_{nullptr};
  ArithmeticOptFunc<float16_t> arithmetic_opt_run_fp16_{nullptr};
  std::vector<void *> fp16_buffer_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_ARITHMETIC_FP16_H_
