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
#include "tools/converter/micro/coder/opcoders/op_coder.h"
#include "tools/converter/micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "nnacl/matmul_parameter.h"
#include "coder/opcoders/nnacl/dynamic_parameter/matmul_dynamic_parameter.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
class MatMulDynamicFP16BaseCoder : public OperatorCoder {
 public:
  MatMulDynamicFP16BaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                             const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~MatMulDynamicFP16BaseCoder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitBiasData(CoderContext *const context);
  int InitMatrixB(CoderContext *const context);
  int ComputeMatrixAWorkspace();
  int CollectFilesForTarget(CoderContext *const context);

 protected:
  virtual int InitAShape() = 0;
  virtual int InitBShape() = 0;

 protected:
  Tensor *filter_tensor_{nullptr};
  Tensor *bias_tensor_{nullptr};
  MicroMatmulParameter params_;
  MatmulDynamicParameter dynamic_params_;
  void *b_pack_ptr_ = nullptr;
  void *bias_ptr_{nullptr};
  int col_tile_{0};
  int row_tile_{0};
  size_t bias_pack_ptr_size_{0};
  size_t b_pack_ptr_size_{0};
  TypeId data_type_{kNumberTypeFloat16};
  int b_batch_{0};
  std::string buffers_start_;
  std::string bias_str_;
  std::string input_a_pack_str_;
  std::string input_b_pack_str_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_MATMUL_FP16_BASE_CODER_H_
