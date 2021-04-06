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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_BASIC_TOKENIZER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_BASIC_TOKENIZER_OP_H_
#include <memory>
#include <string>
#include <unordered_set>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/case_fold_op.h"
#include "minddata/dataset/text/kernels/normalize_utf8_op.h"
#include "minddata/dataset/text/kernels/regex_replace_op.h"
#include "minddata/dataset/text/kernels/regex_tokenizer_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class BasicTokenizerOp : public TensorOp {
 public:
  static const bool kDefLowerCase;
  static const bool kDefKeepWhitespace;
  static const NormalizeForm kDefNormalizationForm;
  static const bool kDefPreserveUnusedToken;
  static const bool kDefWithOffsets;

  explicit BasicTokenizerOp(const bool &lower_case = kDefLowerCase, const bool &keep_whitespace = kDefKeepWhitespace,
                            const NormalizeForm &normalization_form = kDefNormalizationForm,
                            const bool &preserve_unused_token = kDefPreserveUnusedToken,
                            const bool &with_offsets = kDefWithOffsets);

  ~BasicTokenizerOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

 protected:
  Status CaseFoldWithoutUnusedWords(const std::string_view &text, const std::unordered_set<std::string> &unused_words,
                                    std::string *output);
  Status CaseFoldWithoutUnusedWords(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

  std::string Name() const override { return kBasicTokenizerOp; }

 private:
  static const char kCommonPattern[];
  static const char kUnusedPattern[];
  static const std::unordered_set<std::string> kUnusedWords;
  bool with_offsets_;
  bool lower_case_;
  bool keep_whitespace_;
  NormalizeForm normalization_form_;
  bool preserve_unused_token_;
  std::unique_ptr<CaseFoldOp> case_fold_;
  std::unique_ptr<NormalizeUTF8Op> nfd_normalize_;
  std::unique_ptr<NormalizeUTF8Op> common_normalize_;
  std::unique_ptr<RegexReplaceOp> replace_accent_chars_;
  std::unique_ptr<RegexReplaceOp> replace_control_chars_;
  std::unique_ptr<RegexTokenizerOp> regex_tokenizer_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_BASIC_TOKENIZER_OP_H_
