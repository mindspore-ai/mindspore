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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_FP32_BASE_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_FP32_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore::kernel {
class LstmFp32BaseCPUKernel : public LiteKernel {
 public:
  LstmFp32BaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmFp32BaseCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoSequenceLoop(int task_id);

 protected:
  virtual int InitInputWeightBias() = 0;
  virtual int InitStateWeightBias() = 0;
  virtual int InitProjectWeight() = 0;
  virtual void LstmUnidirectional(float *output, const float *weight_h, const float *state_bias, float *hidden_state,
                                  float *cell_state, const float *weight_project, float *intermediate_states,
                                  float *buffer[], bool is_backward) = 0;

  int hidden_init_index_{0};
  int cell_init_index_{0};
  int row_tile_{0};
  int col_tile_{0};
  int state_row_tile_{0};
  int state_col_tile_{0};
  int weight_segment_num_{0};
  float *weight_i_ptr_{nullptr};
  float *weight_h_ptr_{nullptr};
  float *weight_project_ptr_{nullptr};
  float *input_bias_{nullptr};
  float *state_bias_{nullptr};
  LstmParameter *lstm_param_{nullptr};
  std::vector<void *> running_buffer_;

 private:
  void FreeRunBuffer();
  int MallocRunBuffer(bool is_double);
  int ExecuteBidirectionalWithMultiThread();
  int ExecuteUnidirectionalOrSingleThread();
  int LstmPreProcessWithInput(const float *weight_i, const float *input_bias, float *dst);
  void LstmForwardLoop(float *buffer[]);
  void LstmBackwardLoop(float *buffer[]);
  float *packed_input_{nullptr};
  float *intermediate_states_{nullptr};
  float *buffer_forward_[C9NUM] = {nullptr};
  float *buffer_backward_[C9NUM] = {nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_LSTM_FP32_BASE_H_
