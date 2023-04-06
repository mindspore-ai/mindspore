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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CUSTOM_GRU_FP32_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CUSTOM_GRU_FP32_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/custom_gru_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
class CustomGruFP32Coder : public OperatorCoder {
 public:
  CustomGruFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~CustomGruFP32Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 protected:
  virtual void InitNnaclFile(CoderContext *const context);
  virtual void InitPackMatrixB(NNaclFp32Serializer *init_code, const std::string &src, const std::string &dst, int row,
                               int col);
  int data_type_size_{C4NUM};
  int row_tile_{C12NUM};
  int col_tile_{C8NUM};
  void *weight_input_{nullptr};
  void *weight_hidden_{nullptr};
  void *bias_input_{nullptr};
  void *bias_hidden_{nullptr};
  size_t weight_in_pack_size_{0};
  size_t weight_hidden_pack_size_{0};
  size_t bias_pack_size_{0};
  std::string data_type{"float"};
  std::string op_func_{"CustomGru"};
  CustomGruParameter *param_{nullptr};

 private:
  int InitParamter();
  int InitWeightAndBias();
  int ReSize();
  void InitWeightCode(CoderContext *const context, NNaclFp32Serializer *init_code);
  void InitBiasCode(CoderContext *const context, NNaclFp32Serializer *init_code);
  void *run_buffer_{nullptr};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CUSTOM_GRU_FP32_CODER_H_
