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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_AFFINE_FP32_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_AFFINE_FP32_CODER_H_

#include <vector>
#include <string>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/affine_parameter.h"
#include "tools/converter/micro/coder/wrapper/base/affine_wrapper.h"

namespace mindspore::lite::micro::nnacl {
LiteGraph::Node *CreateMatmulNode(void *prim_buf, const std::string &name);
void *CreateMatmulPrimitive();

class AffineFP32Coder final : public OperatorCoder {
 public:
  AffineFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~AffineFP32Coder() override {
    free(matmul_primitive_);
    delete matmul_node_;
    matmul_node_ = nullptr;
  }

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  int ReSize(CoderContext *const context);
  std::string GenSpliceCode();
  int PrepareSpliceOp();
  AffineParameter *affine_param_{nullptr};
  SpliceWrapperParam *splice_param_{nullptr};
  void *matmul_primitive_{nullptr};
  lite::LiteGraph::Node *matmul_node_{nullptr};

  lite::Tensor *full_input_{nullptr};
  lite::Tensor *increment_input_{nullptr};
  lite::Tensor *increment_output_{nullptr};
  lite::Tensor *splice_output_{nullptr};

  std::unique_ptr<OperatorCoder> matmul_coder_{nullptr};
  bool full_run_{true};
  int matmul_col_{0};
  int matmul_row_{0};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_AFFINE_FP32_CODER_H_
