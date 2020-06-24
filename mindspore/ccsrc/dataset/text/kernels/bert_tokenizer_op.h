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
#ifndef DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
#define DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
#include <memory>
#include <string>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/text/kernels/basic_tokenizer_op.h"
#include "dataset/text/kernels/wordpiece_tokenizer_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class BertTokenizerOp : public TensorOp {
 public:
  explicit BertTokenizerOp(const std::shared_ptr<Vocab> &vocab,
                           const std::string &suffix_indicator = WordpieceTokenizerOp::kDefSuffixIndicator,
                           const int &max_bytes_per_token = WordpieceTokenizerOp::kDefMaxBytesPerToken,
                           const std::string &unknown_token = WordpieceTokenizerOp::kDefUnknownToken,
                           bool lower_case = BasicTokenizerOp::kDefLowerCase,
                           bool keep_whitespace = BasicTokenizerOp::kDefKeepWhitespace,
                           NormalizeForm normalization_form = BasicTokenizerOp::kDefNormalizationForm,
                           bool preserve_unused_token = BasicTokenizerOp::kDefPreserveUnusedToken)
      : wordpiece_tokenizer_(vocab, suffix_indicator, max_bytes_per_token, unknown_token),
        basic_tokenizer_(lower_case, keep_whitespace, normalization_form, preserve_unused_token) {}

  ~BertTokenizerOp() override = default;

  void Print(std::ostream &out) const override { out << "BertTokenizerOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  WordpieceTokenizerOp wordpiece_tokenizer_;
  BasicTokenizerOp basic_tokenizer_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
