/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_LSTM_DYNAMIC_FP16_CODER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_LSTM_DYNAMIC_FP16_CODER_H

#include <vector>
#include <string>
#include "nnacl/lstm_parameter.h"
#include "coder/opcoders/nnacl/dynamic_parameter/dynamic_lstm_parameter.h"
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro::nnacl {

class LstmMindirDynamicFP16Coder : public OperatorCoder {
 public:
  LstmMindirDynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                             const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~LstmMindirDynamicFP16Coder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  int InitParam();
  int ComputeWorkSpace();
  void CreateBufferAddrStr();
  int InitInputWeightBias(CoderContext *const context);
  int InitStateWeightBias(CoderContext *const context);
  int InitProjectWeight(CoderContext *const context);
  void GenerateStateWeightBiasStr();
  bool gpu_state_{false};
  TypeId data_type_{kNumberTypeFloat16};
  int weight_segment_num_{0};
  size_t hi_size_{0};
  size_t hh_size_{0};
  size_t hp_size_{0};
  size_t bias_size_{0};
  void *weight_i_ptr_{nullptr};
  void *weight_h_ptr_{nullptr};
  void *weight_project_ptr_{nullptr};
  void *input_bias_{nullptr};
  void *hh_bias_{nullptr};
  void *project_bias_{nullptr};
  std::string weight_i_str_;
  std::string weight_h_str_;
  std::string weight_pro_str_;
  std::string input_bias_str_;
  std::string state_bias_str_;
  std::string pro_bias_str_;
  LstmParameter *lstm_param_{nullptr};
  DynamicLstmParameter dynamic_lstm_param_;
  std::string buffers_start_;
  std::vector<std::string> buffers_str_;
};
}  // namespace mindspore::lite::micro::nnacl

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_LSTM_DYNAMIC_FP16_CODER_H
