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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_CONCAT_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_CONCAT_INT8_CODER_H_

#include <cstring>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/concat_int8.h"

namespace mindspore::lite::micro::nnacl {
class ConcatInt8Coder final : public OperatorCoder {
 public:
  ConcatInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConcatInt8Coder() {
    if (concat_param_ == nullptr) {
      return;
    }
    if (concat_param_->quant_arg_.in_args_ != nullptr) {
      free(concat_param_->quant_arg_.in_args_);
    }
    if (concat_param_->input_shapes_ != nullptr) {
      free(concat_param_->input_shapes_);
    }
  }

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  ConcatParameter *concat_param_{nullptr};
  int64_t before_axis_size{0};
  int64_t count_unit_{0};
  int8_t *input_data_{nullptr};
  int axis_ = 0;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_CONCAT_INT8_CODER_H_
