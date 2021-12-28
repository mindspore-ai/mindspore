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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32_grad/lstm_grad_fp32.h"

namespace mindspore {
namespace kernel {
constexpr int LSTMGRAD_MAX_WORKSPACE_SIZE = 100000;
constexpr int LSTMGRAD_MAX_WEIGHTS_SIZE = 100000;
class LSTMGradCPUKernel : public InnerKernel {
 public:
  explicit LSTMGradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }
  ~LSTMGradCPUKernel() { FreeTmpBuffer(); }
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGrad(int thread_id);

 private:
  int LstmBackpropUnidirectional(float *output, bool is_backward);

  int InitParam();
  void FreeTmpBuffer();
  int MallocRunBuffer();
  void FreeRunBuffer();
  int InitInputWeightBias();
  int InitStateWeightBias();
  float *InputWeightPtr();
  float *StateWeightPtr();
  float *InputBiasPtr();
  float *StateBiasPtr();
  int AllocateWeights();
  int PackWeights();
  void ReorderLstmWeightGrad(float *dst, float *src);

  int thread_count_;
  static const int input_index = 0;
  static const int cell_state_index = 2;
  static const int weights_index = 3;
  static const int dy_index = 7;
  static const int dH_index = 8;
  static const int dC_index = 9;
  static const int intermediate_data_index = 10;
  static const int dX_out_index = 0;
  static const int dH_out_index = 1;
  static const int dC_out_index = 2;
  static const int dW_out_index = 3;
  static const int num_of_gates = 4;
  const int weights_order_IFOG[2 * 4] = {0, 2, 3, 1, 4, 6, 7, 4};  // IFGO order to IOFG order
  const int weights_order_IOFG[2 * 4] = {0, 3, 1, 2, 4, 7, 5, 6};  // IOFG order to IFGO order

  int input_size_align_ = 1;
  float *weight_i_ptr_ = nullptr;
  float *weight_h_ptr_ = nullptr;
  float *input_bias_ = nullptr;
  float *state_bias_ = nullptr;
  float *dW_tmp_ = nullptr;
  float *workspace_ = nullptr;

  int64_t weight_size_ = 0;
  int64_t weight_h_size_ = 0;
  int64_t input_size_;
  int64_t hidden_size_;
  int64_t num_layers_;
  int64_t batch_size_;
  int64_t seq_len_;
  int num_directions_;
  bool bidirectional_;
  bool has_bias_;
  size_t reserve_size_;
  int row_tile_ = 0;
  int col_tile_ = 0;
  int state_row_tile_ = 0;
  int state_col_tile_ = 0;
  int weight_batch_ = 0;
  bool state_is_vec_ = false;
  int input_thread_count_ = 0;
  int input_thread_stride_ = 0;

  LstmParameter *lstm_param_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_LSTM_GRAD_H_
