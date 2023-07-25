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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_NON_MINDIR_FP32_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_NON_MINDIR_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/lstm_fp32_base.h"

namespace mindspore::kernel {
/*
 * 1. LSTM without project, output_size = hidden_size
 *    weight_ih: second input, shape is [bidirectional, 4 * hidden_size, input_size]
 *    weight_hh: third input, shape is [bidirectional, 4 * hidden_size, hidden_size]
 *    bias: forth input, shape is [bidirectional, 8 * hidden_size]
 *    h_init: fifth input, shape is [bidirectional, batch_size, hidden_size]
 *    c_init: sixth input, shape is [bidirectional, batch_size, hidden_size]
 *
 * 2. LSTM with project, output_size = project_size
 *    weight_ih: second input, shape is [bidirectional, 4 * hidden_size, input_size]
 *    weight_hh: third input, shape is [bidirectional, 4 * hidden_size, project_size]
 *    bias: forth input, shape is [bidirectional, 8 * hidden_size]
 *    h_init: fifth input, shape is [bidirectional, batch_size, project_size]
 *    c_init: sixth input, shape is [bidirectional, batch_size, hidden_size]
 *    weight_pro: seventh input, shape is [bidirectional, project_size, hidden_size]
 */
class LstmNonMindirFp32CPUKernel : public LstmFp32BaseCPUKernel {
 public:
  LstmNonMindirFp32CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LstmFp32BaseCPUKernel(parameter, inputs, outputs, ctx) {
    hidden_init_index_ = FIFTH_INPUT;
    cell_init_index_ = SIXTH_INPUT;
  }

  ~LstmNonMindirFp32CPUKernel() override = default;

 protected:
  int InitInputWeightBias() override;
  int InitStateWeightBias() override;
  int InitProjectWeight() override;
  void LstmUnidirectional(float *output, const float *weight_h, const float *state_bias, float *hidden_state,
                          float *cell_state, const float *weight_project, float *intermediate_states, float *buffer[],
                          bool is_backward) override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_NON_MINDIR_FP32_H_
