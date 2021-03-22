/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_LSTM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_LSTM_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/lstm_parameter.h"

namespace mindspore::kernel {
class LstmFp16CPUKernel : public LiteKernel {
 public:
  LstmFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmFp16CPUKernel() override { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  void FreeTmpBuffer();
  void FreeRunBuffer();
  int InitParam();
  int InitInputWeightBias();
  int InitStateWeightBias();
  int MallocRunBuffer();

  float16_t *weight_i_ptr_ = nullptr;
  float16_t *weight_h_ptr_ = nullptr;
  float16_t *input_bias_ = nullptr;
  float16_t *state_bias_ = nullptr;
  float16_t *buffer_[6];
  int weight_batch_ = 0;
  bool is_vec_ = false;
  LstmParameter *lstm_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_LSTM_H_
