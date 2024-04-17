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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_DYNAMIC_BASE_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_DYNAMIC_BASE_CODER_H_
#include <vector>
#include <string>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/nnacl/dynamic_parameter/strided_slice_dynamic_parameter.h"
#include "nnacl/strided_slice_parameter.h"
#include "nnacl/kernel/strided_slice.h"

namespace mindspore::lite::micro {
class StridedSliceDynamicBaseCoder final : public OperatorCoder {
 public:
  StridedSliceDynamicBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                               const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~StridedSliceDynamicBaseCoder() override = default;

  int Prepare(CoderContext *context) override;

  int DoCode(CoderContext *context) override;

  void PadStridedSliceParamTo8D();

 private:
  StridedSliceParameter *strided_slice_param_{nullptr};
  StridedSliceStruct struct_;
  StridedSliceDynamicParameter dynamic_param_;
  size_t inner_{1};
  size_t inner_size_{1};
  std::vector<std::string> end_;
  std::vector<std::string> input_shape_;
  std::vector<std::string> output_shape_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_DYNAMIC_BASE_CODER_H_
