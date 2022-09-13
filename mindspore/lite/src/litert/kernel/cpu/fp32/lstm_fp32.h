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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_LSTM_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_LSTM_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore::kernel {
class LstmCPUKernel : public LiteKernel {
 public:
  LstmCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

  void InputWeightMatMul(int task_id) const;
  int DoSequenceLoop(int task_id);

 private:
  void FreeRunBuffer();
  int InitParam();
  int MallocRunBuffer(bool is_double);
  int InitInputWeightBias();
  int InitStateWeightBias();
  int LstmPreProcessWithInput(const float *weight_i, const float *input_bias, float *dst);
  int ExecuteUnidirectionalOrSingleThread();
  int ExecuteBidirectionalWithMultiThread();
  void LstmForwardLoop(float *buffer[]);
  void LstmBackwardLoop(float *buffer[]);
  void LstmUnidirectional(float *output, const float *weight_h, const float *state_bias, float *hidden_state,
                          float *cell_state, float *intermediate_states, float *buffer[], bool is_backward);
  void RecordStates(const float *hidden_state, float *cell_state, float *input_gate, const float *output_gate,
                    float *forget_gate, const float *cell_gate, float *intermediate_states, int step);
  const float *weight_loop_;
  const float *bias_loop_;
  float *gate_loop_ = nullptr;
  int input_thread_count_ = 0;
  int input_thread_stride_ = 0;

  float *weight_i_ptr_ = nullptr;
  float *weight_h_ptr_ = nullptr;
  float *input_bias_ = nullptr;
  float *state_bias_ = nullptr;
  float *intermediate_states_ = nullptr;
  // indices of weights when split
  const size_t mindir_input_tensors = 4;
  const int onnx_weight_i_index = 1;
  const int onnx_weight_h_index = 2;
  const int onnx_bias_index = 3;
  const int onnx_hidden_state_index = 4;
  const int onnx_cell_state_index = 5;
  // index of combined weightes when combined
  const int combined_weights_index = 3;
  const int mindir_hidden_state_input_index = 1;
  const int mindir_cell_state_input_index = 2;
  int hidden_state_input_index_ = onnx_hidden_state_index;
  int cell_state_input_index_ = onnx_cell_state_index;

  float *packed_input_{nullptr};
  float *buffer_forward_[C7NUM] = {nullptr};
  float *buffer_backward_[C7NUM] = {nullptr};
  std::vector<void *> buffer_running_malloc_;
  const int gate_num = 4;
  const int input_gate_index = 0;
  const int tmp_hidden_output_index = 6;
  static const int out_intermediate_states_index = 3;
  const int weights_order_IFOG[2 * 4] = {0, 2, 3, 1, 4, 6, 7, 5};  // IFGO order to IOFG order

  int row_tile_ = 0;
  int col_tile_ = 0;
  int state_row_tile_ = 0;
  int state_col_tile_ = 0;
  int weight_batch_ = 0;
  bool state_is_vec_ = false;
  // control weight layout
  bool gpu_orig_state_ = true;
  bool gpu_orig_cfg_ = true;
  LstmParameter *lstm_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_LSTM_FP32_H_
