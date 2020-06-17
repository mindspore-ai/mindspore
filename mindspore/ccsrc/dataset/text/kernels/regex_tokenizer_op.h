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
#ifndef DATASET_TEXT_REGEX_TOKENIZER_OP_H_
#define DATASET_TEXT_REGEX_TOKENIZER_OP_H_
#include <memory>
#include <string>
#include <vector>

#include "unicode/regex.h"
#include "unicode/errorcode.h"
#include "unicode/utypes.h"

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {

class RegexTokenizerOp : public TensorOp {
 public:
  RegexTokenizerOp(const std::string &delim_pattern, const std::string &keep_delim_pattern)
      : delim_pattern_(icu::UnicodeString::fromUTF8(delim_pattern)),
        keep_delim_pattern_(icu::UnicodeString::fromUTF8(keep_delim_pattern)),
        keep_delim_(!keep_delim_pattern.empty()) {}

  ~RegexTokenizerOp() override = default;

  void Print(std::ostream &out) const override { out << "RegexTokenizerOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 protected:
  Status GetUnicodeSubstr(const icu::UnicodeString &input, int start, int len, std::string *out_utf8,
                          icu::UnicodeString *out_unicode = nullptr) const;
  Status GetRegexTokens(const std::string &text, std::vector<std::string> *out_tokens) const;

 private:
  const icu::UnicodeString delim_pattern_;
  const icu::UnicodeString keep_delim_pattern_;
  const bool keep_delim_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_TEXT_REGEX_TOKENIZER_OP_H_
