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

#ifndef MINDSPORE_LITE_MICRO_OPCODERS_NNACL_FP32_CONV2D_DELEGATE_FP32_CODER_H
#define MINDSPORE_LITE_MICRO_OPCODERS_NNACL_FP32_CONV2D_DELEGATE_FP32_CODER_H
#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/conv_parameter.h"
namespace mindspore::lite::micro::nnacl {
class ConvDelegateCoder : public OperatorCoder {
 public:
  ConvDelegateCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvDelegateCoder() override = default;
  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 protected:
  std::unique_ptr<OperatorCoder> conv_coder_ = nullptr;
};

void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output);
std::unique_ptr<OperatorCoder> CPUConvolutionFP32CoderSelect(const std::vector<Tensor *> &in_tensors,
                                                             const std::vector<Tensor *> &out_tensors,
                                                             const Model::Node *node, size_t node_index, Target target);
std::unique_ptr<OperatorCoder> CreateDelegateConv(const std::vector<Tensor *> &in_tensors,
                                                  const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                                  size_t node_index, Target target);
std::unique_ptr<OperatorCoder> CPUConvDwFp32CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                         const std::vector<Tensor *> &out_tensors,
                                                         const Model::Node *node, size_t node_index, Target target);

std::unique_ptr<OperatorCoder> CPUConv2DFusionFP32CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                               const std::vector<Tensor *> &out_tensors,
                                                               const Model::Node *node, size_t node_index,
                                                               Target target);

}  // namespace mindspore::lite::micro::nnacl

#endif  // MINDSPORE_LITE_MICRO_OPCODERS_NNACL_FP32_CONV2D_DELEGATE_FP32_CODER_H
