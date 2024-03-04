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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_DYNAMIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_DYNAMIC_FP16_CODER_H_

#include <vector>
#include <string>
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/nnacl/dynamic_parameter/conv_dynamic_parameter.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
class Convolution1x1DynamicFP16Coder final : public OperatorCoder {
 public:
  Convolution1x1DynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                 const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~Convolution1x1DynamicFP16Coder() override;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  void CollectFilesForFunc(CoderContext *const context);
  int InitWeightBias(CoderContext *const context);
  int InitMatmulParam();
  int InitTmpBuffer(CoderContext *const context);
  void FreeTmpBuffer();
  int ComputeWorkspace();
  MatMulParameter *matmul_param_{nullptr};
  ConvParameter *conv_param_{nullptr};
  ConvDynamicParameter dynamic_param_;
  Tensor *filter_tensor_{nullptr};
  Tensor *bias_tensor_{nullptr};
  int row_tile_{C12NUM};
  int col_tile_{C8NUM};
  void *packed_weight_{nullptr};
  void *bias_data_{nullptr};
  std::string pack_input_str_;
  void *tmp_input_{nullptr};
  size_t pack_weight_size_{0};
  size_t bias_data_size_{0};
  size_t tmp_input_size_{0};
  size_t pack_input_size_{0};
  bool pre_trans_input_{false};
  std::string output_ptr_;
  TypeId data_type_ = kNumberTypeFloat16;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_DYNAMIC_FP16_CODER_H_
