/**
 * Copyright 2023Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_MINDIR_FP32_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_MINDIR_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/lstm_fp32_base.h"

namespace mindspore::kernel {
/*
 * 1. LSTM without project
 *    h_init: second input, shape is [bidirectional, batch_size, hidden_size]
 *    c_init: third input, shape is [bidirectional, batch_size, hidden_size]
 *    weight_bias: forth input, weight_ih + weight_hh + bias, the gate order is IFGO
 *
 * 2. LSTM with project
 *    don't support
 *    h_init: second input, shape is [bidirectional, batch_size, hidden_size]
 *    c_init: third input, shape is [bidirectional, batch_size, hidden_size]
 *    weight_bias: forth input, weight_ih + weight_hh + proj + bias, the gate order is IFGO
 */
class LstmMindirFp32CPUKernel : public LstmFp32BaseCPUKernel {
 public:
  LstmMindirFp32CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LstmFp32BaseCPUKernel(parameter, inputs, outputs, ctx) {
    hidden_init_index_ = SECOND_INPUT;
    cell_init_index_ = THIRD_INPUT;
  }

  ~LstmMindirFp32CPUKernel() override = default;

  int ReSize() override;

 protected:
  int InitInputWeightBias() override;
  int InitStateWeightBias() override;
  int InitProjectWeight() override;
  void LstmUnidirectional(float *output, const float *weight_h, const float *state_bias, float *hidden_state,
                          float *cell_state, const float *weight_project, float *intermediate_states, float *buffer[],
                          bool is_backward) override;

 private:
  void RecordStates(const float *hidden_state, float *cell_state, float *input_gate, const float *output_gate,
                    float *forget_gate, const float *cell_gate, float *intermediate_states, int step);
  bool gpu_orig_state_{false};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_MINDIR_FP32_H_
