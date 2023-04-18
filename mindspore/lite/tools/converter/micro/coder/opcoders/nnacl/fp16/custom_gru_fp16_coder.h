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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CUSTOM_GRU_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CUSTOM_GRU_FP16_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/nnacl/fp32/custom_gru_fp32_coder.h"
#include "nnacl/custom_gru_parameter.h"

namespace mindspore::lite::micro::nnacl {
class CustomGruFP16Coder : public CustomGruFP32Coder {
 public:
  CustomGruFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const LiteGraph::Node *node, size_t node_index, Target target)
      : CustomGruFP32Coder(in_tensors, out_tensors, node, node_index, target) {
    data_type_size_ = sizeof(uint16_t);
    row_tile_ = C4NUM;
    col_tile_ = C8NUM;
    data_type = "float16_t";
    op_func_ = "CustomGruFp16";
  }
  ~CustomGruFP16Coder() override = default;

 protected:
  void InitNnaclFile(CoderContext *const context) override;
  void InitPackMatrixB(NNaclFp32Serializer *init_code, const std::string &src, const std::string &dst, int row,
                       int col) override;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CUSTOM_GRU_FP16_CODER_H_
