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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_BASE_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_BASE_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/nnacl/fp32/matmul_fp32_base_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite::micro::nnacl {
class MatMulFP16BaseCoder : public MatMulFP32BaseCoder {
 public:
  MatMulFP16BaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const LiteGraph::Node *node, size_t node_index, Target target)
      : MatMulFP32BaseCoder(in_tensors, out_tensors, node, node_index, target) {
    data_type_ = kNumberTypeFloat16;
  }

  ~MatMulFP16BaseCoder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitBufferForBias() override;
  std::string InitBiasData(NNaclFp32Serializer *const init_code, CoderContext *const context, size_t *w_buf);
  std::string InitMatrixA(NNaclFp32Serializer *const code, NNaclFp32Serializer *const init_code,
                          CoderContext *const context, size_t *w_buf);
  std::string InitMatrixB(NNaclFp32Serializer *const code, NNaclFp32Serializer *const init_code,
                          CoderContext *const context, size_t *w_buf);
  int CollectFilesForTarget(CoderContext *const context) override;

 protected:
  virtual int InitAShape() = 0;
  virtual int InitBShape() = 0;
  int InitBufferA() override;
  int InitBufferB() override;

 protected:
  int a_batch_ = 1;
  int b_batch_ = 1;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_BASE_CODER_H_
