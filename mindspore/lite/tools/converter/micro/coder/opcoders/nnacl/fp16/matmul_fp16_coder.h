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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_CODER_H_

#include <vector>
#include "coder/opcoders/nnacl/fp16/matmul_fp16_base_coder.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite::micro::nnacl {
class MatMulFP16Coder final : public MatMulFP16BaseCoder {
 public:
  MatMulFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const LiteGraph::Node *node, size_t node_index, Target target)
      : MatMulFP16BaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~MatMulFP16Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitAShape() override;
  int InitBShape() override;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_CODER_H_
