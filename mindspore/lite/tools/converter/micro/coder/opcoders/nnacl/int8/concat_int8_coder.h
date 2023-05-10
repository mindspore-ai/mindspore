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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_CONCAT_INT8_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_CONCAT_INT8_CODER_H_

#include <cstring>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/concat_int8.h"
#include "wrapper/int8/concat_int8_wrapper.h"

namespace mindspore::lite::micro::nnacl {
class ConcatInt8Coder final : public OperatorCoder {
 public:
  ConcatInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConcatInt8Coder() {
    if (concat_param_ == nullptr) {
      return;
    }
    if (concat_param_->quant_arg_.in_args_ != nullptr) {
      free(concat_param_->quant_arg_.in_args_);
    }
    if (micro_concat_.input_shapes_ != nullptr) {
      free(micro_concat_.input_shapes_);
    }
  }

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  ConcatParameter *concat_param_{nullptr};
  ConcatInt8Args micro_concat_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_CONCAT_INT8_CODER_H_
