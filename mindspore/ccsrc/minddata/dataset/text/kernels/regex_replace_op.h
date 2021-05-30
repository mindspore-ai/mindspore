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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
#include <memory>
#include <string>

#include "unicode/regex.h"
#include "unicode/errorcode.h"
#include "unicode/utypes.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class RegexReplaceOp : public TensorOp {
 public:
  RegexReplaceOp(const std::string &pattern, const std::string &replace, bool replace_all = true)
      : pattern_(icu::UnicodeString::fromUTF8(pattern)),
        replace_(icu::UnicodeString::fromUTF8(replace)),
        replace_all_(replace_all) {}

  ~RegexReplaceOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRegexReplaceOp; }

 protected:
  Status RegexReplace(icu::RegexMatcher *const matcher, const std::string_view &text, std::string *out) const;

 private:
  const icu::UnicodeString pattern_;
  const icu::UnicodeString replace_;
  const bool replace_all_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
