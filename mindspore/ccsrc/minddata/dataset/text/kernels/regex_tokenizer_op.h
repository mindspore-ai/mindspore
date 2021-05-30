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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_REGEX_TOKENIZER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_REGEX_TOKENIZER_OP_H_
#include <memory>
#include <string>
#include <vector>

#include "unicode/regex.h"
#include "unicode/errorcode.h"
#include "unicode/utypes.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/tokenizer_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class RegexTokenizerOp : public TokenizerOp {
 public:
  RegexTokenizerOp(const std::string &delim_pattern, const std::string &keep_delim_pattern,
                   const bool &with_offsets = kDefWithOffsets)
      : TokenizerOp(with_offsets),
        delim_pattern_(icu::UnicodeString::fromUTF8(delim_pattern)),
        keep_delim_pattern_(icu::UnicodeString::fromUTF8(keep_delim_pattern)),
        keep_delim_(!keep_delim_pattern.empty()) {}

  ~RegexTokenizerOp() override = default;

  Status Tokenize(std::string_view str, std::vector<std::string> *splits, std::vector<uint32_t> *offsets_start,
                  std::vector<uint32_t> *offsets_limit) override;

 protected:
  Status GetUnicodeSubstr(const icu::UnicodeString &input, const int &start, const int &len, std::string *out_utf8,
                          icu::UnicodeString *out_unicode = nullptr) const;
  Status GetRegexTokens(const std::string &text, std::vector<std::string> *out_tokens,
                        std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) const;

  std::string Name() const override { return kRegexTokenizerOp; }

 private:
  const icu::UnicodeString delim_pattern_;
  const icu::UnicodeString keep_delim_pattern_;
  const bool keep_delim_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_REGEX_TOKENIZER_OP_H_
