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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_SOFTMAX_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_SOFTMAX_INT8_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/int8/quantize.h"

namespace mindspore::kernel {
class SoftmaxInt8CPUKernel : public LiteKernel {
 public:
  SoftmaxInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    softmax_param_ = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  }
  ~SoftmaxInt8CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoSoftmax(int task_id);

 private:
  int n_dim_;
  int element_size_;
  int input_shape_[DIMENSION_5D];
  int *sum_data_ = nullptr;
  int *exp_data_ = nullptr;
  SoftmaxParameter *softmax_param_;
  SoftmaxQuantArg *quant_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_SOFTMAX_INT8_H_
