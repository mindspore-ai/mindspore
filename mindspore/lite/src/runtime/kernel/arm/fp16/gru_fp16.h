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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRU_H_
#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/gru_parameter.h"

namespace mindspore::kernel {
class GruFp16CPUKernel : public InnerKernel {
 public:
  GruFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    gru_param_ = reinterpret_cast<GruParameter *>(op_parameter_);
  }

  ~GruFp16CPUKernel() override { FreeTmpBuffer(); }

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

  float16_t *weight_g_ptr_ = nullptr;
  float16_t *weight_r_ptr_ = nullptr;
  float16_t *input_bias_ = nullptr;
  float16_t *state_bias_ = nullptr;

  float16_t *buffer_[4] = {0};
  const int gate_num = 3;
  const int packed_input_index = 0;
  const int input_gate_index = 1;
  const int packed_state_index = 2;
  const int state_gate_index = 3;

  int weight_batch_ = 0;
  bool is_vec_ = false;
  GruParameter *gru_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRU_H_
