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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_UNICODE_SCRIPT_TOKENIZER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_UNICODE_SCRIPT_TOKENIZER_OP_H_
#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/tokenizer_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class UnicodeScriptTokenizerOp : public TokenizerOp {
 public:
  static const bool kDefKeepWhitespace;

  explicit UnicodeScriptTokenizerOp(const bool &keep_whitespace = kDefKeepWhitespace,
                                    const bool &with_offsets = kDefWithOffsets)
      : TokenizerOp(with_offsets), keep_whitespace_(keep_whitespace) {}

  ~UnicodeScriptTokenizerOp() override = default;

  Status Tokenize(std::string_view str, std::vector<std::string> *splits, std::vector<uint32_t> *offsets_start,
                  std::vector<uint32_t> *offsets_limit) override;

  std::string Name() const override { return kUnicodeScriptTokenizerOp; }

 private:
  bool keep_whitespace_;  // If or not keep whitespace tokens
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_UNICODE_SCRIPT_TOKENIZER_OP_H_
