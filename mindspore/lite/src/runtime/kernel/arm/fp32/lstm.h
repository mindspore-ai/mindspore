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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/nnacl/fp32/lstm.h"

namespace mindspore::kernel {
class LstmCPUKernel : public LiteKernel {
 public:
  LstmCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    lstm_parm_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmCPUKernel() override { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;

  int InitParam();
  int InitBuffer();
  int InitWeightBias();

 private:
  void FreeTmpBuffer() {
    if (gate_buffer_ != nullptr) {
      free(gate_buffer_);
      gate_buffer_ = nullptr;
    }
    if (weight_i_ptr_ != nullptr) {
      free(weight_i_ptr_);
      weight_i_ptr_ = nullptr;
    }
    if (weight_h_ptr_ != nullptr) {
      free(weight_h_ptr_);
      weight_h_ptr_ = nullptr;
    }
    if (bias_ptr_ != nullptr) {
      free(bias_ptr_);
      bias_ptr_ = nullptr;
    }
  }
  float *gate_buffer_ = nullptr;
  float *weight_i_ptr_ = nullptr;
  float *weight_h_ptr_ = nullptr;
  float *bias_ptr_ = nullptr;
  LstmParameter *lstm_parm_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_H_
