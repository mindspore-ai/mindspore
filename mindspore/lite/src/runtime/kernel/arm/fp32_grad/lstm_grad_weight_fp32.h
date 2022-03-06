/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_WEIGHT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_WEIGHT_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32_grad/lstm_grad_fp32.h"

namespace mindspore {
namespace kernel {
constexpr int LSTMGRADWEIGHT_MAX_WORKSPACE_SIZE = 100000;
constexpr int LSTMGRADWEIGHT_MAX_WEIGHTS_SIZE = 100000;
class LSTMGradWeightCPUKernel : public InnerKernel {
 public:
  explicit LSTMGradWeightCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmGradParameter *>(op_parameter_);
  }
  ~LSTMGradWeightCPUKernel() {}
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGrad(int thread_id);

 private:
  int LstmBackpropUnidirectional(float *output, bool is_backward);

  int InitParam();
  int MallocRunBuffer();
  void FreeRunBuffer();
  void ReorderLstmWeightGrad(float *dst, float *src, bool has_bias);

  static const int input_index = 0;
  static const int hidden_input_index = 1;
  static const int y_index = 2;
  static const int intermediate_data_index = 3;
  static const int dW_out_index = 0;
  static const int num_of_gates = 4;

  int input_size_align_ = 1;
  float *dW_tmp_ = nullptr;
  float *workspace_ = nullptr;

  int row_tile_ = 0;
  int col_tile_ = 0;
  int state_row_tile_ = 0;
  int state_col_tile_ = 0;
  int weight_batch_ = 0;
  bool state_is_vec_ = false;
  int input_thread_count_ = 0;
  int input_thread_stride_ = 0;

  LstmGradParameter *lstm_param_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_WEIGHT_H_
