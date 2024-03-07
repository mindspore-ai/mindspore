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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_SOFTMAX_DYNAMIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_SOFTMAX_DYNAMIC_FP16_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/nnacl/dynamic_parameter/softmax_dynamic_parameter.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/kernel/softmax.h"

namespace mindspore::lite::micro::nnacl {
class SoftmaxDynamicFP16Coder final : public OperatorCoder {
 public:
  SoftmaxDynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                          const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~SoftmaxDynamicFP16Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int Init();
  int ComputeWorkSpace();
  SoftmaxParameter *param_{nullptr};
  SoftmaxStruct softmax_struct_;
  SoftmaxDynamicParameter dynamic_param_;
  std::vector<std::string> input_shape_;
  std::string buffer_start_;
  std::string sum_data_str_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_SOFTMAX_DYNAMIC_FP16_CODER_H_
