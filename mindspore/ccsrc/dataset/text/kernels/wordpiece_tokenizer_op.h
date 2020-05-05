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
#ifndef DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
#define DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cppjieba/Unicode.hpp"

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/text/vocab.h"
#include "dataset/util/status.h"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;
namespace mindspore {
namespace dataset {

class WordpieceTokenizerOp : public TensorOp {
 public:
  static const char kDefSuffixIndicator[];
  static const int kDefMaxBytesPerToken;
  static const char kDefUnknownToken[];
  WordpieceTokenizerOp(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = kDefSuffixIndicator,
                       const int &max_bytes_per_token = kDefMaxBytesPerToken,
                       const std::string &unknown_token = kDefUnknownToken);

  ~WordpieceTokenizerOp() override = default;

  void Print(std::ostream &out) const override { out << "WordpieceTokenizerOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 protected:
  void PadTokens(const std::vector<std::vector<std::string>> &tokens, const std::string &padded_str,
                 std::vector<std::string> *out_padded_tokens, int *out_cols) const;
  Status AddSubword(const std::string &input_token, const int start, const int end,
                    std::vector<std::string> *out_token) const;
  Status FoundNoToken(const std::string &input_token, std::vector<std::string> *out_tokens) const;
  Status LookupWord(const std::string &input_token, const RuneStrArray &runes, const int start, bool *out_found,
                    int *out_end) const;
  Status GetTokens(const std::string &input_token, std::vector<std::string> *out_tokens) const;

 private:
  const std::shared_ptr<Vocab> vocab_;
  const std::string suffix_indicator_;
  const int max_bytes_per_token_;
  const std::string unknown_token_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
