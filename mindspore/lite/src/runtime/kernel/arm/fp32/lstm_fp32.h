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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_FP32_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore::kernel {
class LstmCPUKernel : public InnerKernel {
 public:
  LstmCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmCPUKernel() override { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;

  void InputWeightMatMul(int task_id);

 private:
  void FreeTmpBuffer();
  void FreeRunBuffer();
  int InitParam();
  int MallocRunBuffer();
  int InitInputWeightBias();
  int InitStateWeightBias();

  int LstmUnidirectional(float *output, const float *weight_i, const float *weight_h, const float *input_bias,
                         const float *state_bias, float *hidden_state, float *cell_state, bool is_backward);
  int InnerExecute(float *output, const float *input, float *hidden_state, float *cell_state);
  void RecordStates(const float *cell_state, int step);
  const float *weight_loop_;
  const float *bias_loop_;
  float *gate_loop_;
  int input_thread_count_ = 0;
  int input_thread_stride_ = 0;

  float *weight_i_ptr_ = nullptr;
  float *weight_h_ptr_ = nullptr;
  float *input_bias_ = nullptr;
  float *state_bias_ = nullptr;
  const int weight_i_index = 1;
  const int weight_h_index = 2;
  const int bias_index = 3;

  float *buffer_[7];
  const int gate_num = 4;
  const int packed_input_index = 0;
  const int input_gate_index = 1;
  const int packed_state_index = 2;
  const int state_gate_index = 3;
  const int cell_state_index = 4;
  const int hidden_state_index = 5;
  const int avx_state_output_index = 6;

  int row_tile_ = 0;
  int col_tile_ = 0;
  int state_row_tile_ = 0;
  int state_col_tile_ = 0;
  int weight_batch_ = 0;
  bool state_is_vec_ = false;
  bool output_need_packed_ = false;
  LstmParameter *lstm_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSTM_FP32_H_
