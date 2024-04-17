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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV2D_DELEGATE_DYNAMIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV2D_DELEGATE_DYNAMIC_FP16_CODER_H_
#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/nnacl/dynamic_parameter/conv_dynamic_parameter.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::micro::nnacl {
class ConvDelegateDynamicFP16Coder : public OperatorCoder {
 public:
  ConvDelegateDynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                               const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvDelegateDynamicFP16Coder() override = default;
  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 protected:
  std::unique_ptr<OperatorCoder> conv_coder_ = nullptr;
  ConvParameter *conv_param_{nullptr};
  ConvDynamicParameter dynamic_param_;

 private:
  void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output);
};

std::unique_ptr<OperatorCoder> CPUConvFP16DynamicCoderSelect(const std::vector<lite::Tensor *> &in_tensors,
                                                             const std::vector<lite::Tensor *> &out_tensors,
                                                             const LiteGraph::Node *node, size_t node_index,
                                                             Target target, int schema_version);

std::unique_ptr<OperatorCoder> CPUConv2DFusionDynamicFP16CoderCreator(const std::vector<lite::Tensor *> &in_tensors,
                                                                      const std::vector<lite::Tensor *> &out_tensors,
                                                                      const LiteGraph::Node *node, size_t node_index,
                                                                      Target target, int schema_version);
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV2D_DELEGATE_DYNAMIC_FP16_CODER_H_
