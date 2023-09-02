/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_MINDIR_FP16_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_MINDIR_FP16_H_

#include <vector>
#include "src/litert/kernel/cpu/fp16/lstm_fp16_base.h"

namespace mindspore::kernel {
/*
 * 1. LSTM without project, output_size = hidden_size
 *    h_init: second input, shape is [bidirectional, batch_size, hidden_size]
 *    c_init: third input, shape is [bidirectional, batch_size, hidden_size]
 *    weight_bias: forth input, weight_ih + weight_hh + bias, the gate order is IFGO
 *
 * 2. LSTM with project, output_size = project_size
 *    don't support
 *    h_init: second input, shape is [bidirectional, batch_size, hidden_size]
 *    c_init: third input, shape is [bidirectional, batch_size, hidden_size]
 *    weight_bias: forth input, weight_ih + weight_hh + proj + bias, the gate order is IFGO
 */
class LstmMindirFp16CPUKernel : public LstmFp16BaseCPUKernel {
 public:
  LstmMindirFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LstmFp16BaseCPUKernel(parameter, inputs, outputs, ctx) {
    hidden_init_index_ = SECOND_INPUT;
    cell_init_index_ = THIRD_INPUT;
  }

  ~LstmMindirFp16CPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;

 protected:
  int InitInputWeightBias() override;
  int InitStateWeightBias() override;
  int InitProjectWeight() override;

 private:
  bool gpu_orig_state_{false};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_MINDIR_FP16_H_
