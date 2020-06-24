/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
#define DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
#include <memory>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {

class UnicodeCharTokenizerOp : public TensorOp {
 public:
  UnicodeCharTokenizerOp() {}

  ~UnicodeCharTokenizerOp() override = default;

  void Print(std::ostream &out) const override { out << "UnicodeCharTokenizerOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
