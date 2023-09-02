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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_FP16_BASE_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_FP16_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/lstm_parameter.h"

namespace mindspore::kernel {
class LstmFp16BaseCPUKernel : public LiteKernel {
 public:
  LstmFp16BaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    lstm_param_ = reinterpret_cast<LstmParameter *>(op_parameter_);
  }

  ~LstmFp16BaseCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 protected:
  virtual int InitInputWeightBias() = 0;
  virtual int InitStateWeightBias() = 0;
  virtual int InitProjectWeight() = 0;
  int InitParam();
  int PackWeightAndBias();
  int PackInputWeight(const void *src, const int32_t *order, TypeId src_data_type);
  int PackInputBias(const void *src, const int32_t *order, TypeId src_data_type);
  int PackStateWeight(const void *src, const int32_t *order, TypeId src_data_type);
  int PackStateBias(const void *src, const int32_t *order, TypeId src_data_type);
  int PackProjectWeight(const void *src, const int32_t *order, TypeId src_data_type);

  bool running_pack_{false};
  bool weight_need_pack_{false};
  int hidden_init_index_{0};
  int cell_init_index_{0};
  int weight_segment_num_{0};
  float16_t *weight_i_ptr_{nullptr};
  float16_t *weight_h_ptr_{nullptr};
  float16_t *weight_project_ptr_{nullptr};
  float16_t *input_bias_{nullptr};
  float16_t *state_bias_{nullptr};
  float16_t *project_bias_{nullptr};
  LstmParameter *lstm_param_{nullptr};
  float16_t *running_buffer_[C7NUM] = {nullptr};
  std::vector<void *> pack_buffer_;

 private:
  void FreePackBuffer();
  void FreeRunBuffer();
  int MallocRunBuffer();
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP16_LSTM_FP16_BASE_H_
