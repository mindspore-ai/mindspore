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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_DYNAMIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_DYNAMIC_FP16_CODER_H_

#include <vector>
#include <string>
#include "nnacl/conv_parameter.h"
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/nnacl/dynamic_parameter/conv_dynamic_parameter.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionDynamicFP16Coder final : public OperatorCoder {
 public:
  ConvolutionDynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                              const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvolutionDynamicFP16Coder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  void CollectFilesForFunc(CoderContext *const context);
  int InitWeightBias(CoderContext *const context);
  int InitTmpBuffer();
  ConvParameter *conv_param_{nullptr};
  ConvDynamicParameter dynamic_param_;
  TypeId data_type_{kNumberTypeFloat16};
  int row_tile_{C12NUM};
  int col_tile_{C8NUM};
  Tensor *filter_tensor_{nullptr};
  Tensor *bias_tensor_{nullptr};
  size_t pack_weight_size_{0};
  size_t packed_input_size_{0};
  void *packed_weight_{nullptr};
  void *bias_data_{nullptr};
  std::string packed_input_str_;
  std::string col_major_input_str_;
  std::string bias_data_str_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_DYNAMIC_FP16_CODER_H_
