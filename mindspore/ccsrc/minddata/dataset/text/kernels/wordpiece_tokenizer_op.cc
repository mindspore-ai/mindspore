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

#include "minddata/dataset/text/kernels/wordpiece_tokenizer_op.h"
#include <algorithm>
#include <utility>

namespace mindspore {
namespace dataset {

const char WordpieceTokenizerOp::kDefSuffixIndicator[] = "##";
const int WordpieceTokenizerOp::kDefMaxBytesPerToken = 100;
const char WordpieceTokenizerOp::kDefUnknownToken[] = "[UNK]";
const bool WordpieceTokenizerOp::kDefWithOffsets = false;

WordpieceTokenizerOp::WordpieceTokenizerOp(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator,
                                           const int &max_bytes_per_token, const std::string &unknown_token,
                                           const bool &with_offsets)
    : vocab_(vocab),
      suffix_indicator_(suffix_indicator),
      max_bytes_per_token_(max_bytes_per_token),
      unknown_token_(unknown_token),
      with_offsets_(with_offsets) {}

Status WordpieceTokenizerOp::LookupWord(const std::string &input_token, const RuneStrArray &runes, const int start,
                                        bool *out_found, int *out_end) const {
  CHECK_FAIL_RETURN_UNEXPECTED(start >= 0 && start < input_token.size(), "WordpieceTokenizer: LookupWord Out of range");
  *out_found = false;
  for (int i = runes.size() - 1; i >= 0; i--) {
    *out_end = runes[i].offset + runes[i].len;
    int len = *out_end - start;
    std::string word = input_token.substr(start, len);
    if (start > 0) {
      word = suffix_indicator_ + word;
    }
    if (vocab_->Lookup(word) != Vocab::kNoTokenExists) {
      *out_found = true;
      break;
    }
  }
  return Status::OK();
}

Status WordpieceTokenizerOp::FoundNoToken(const std::string &input_token, const uint32_t &basic_start,
                                          std::vector<std::string> *out_tokens, std::vector<uint32_t> *offsets_start,
                                          std::vector<uint32_t> *offsets_limit) const {
  out_tokens->clear();
  offsets_start->push_back(basic_start);
  if (unknown_token_.empty()) {
    out_tokens->emplace_back(input_token);
    offsets_limit->push_back(basic_start + input_token.length());
  } else {
    out_tokens->emplace_back(unknown_token_);
    offsets_limit->push_back(basic_start + input_token.length());
  }
  return Status::OK();
}

Status WordpieceTokenizerOp::AddSubword(const std::string &input_token, const int &start, const int &end,
                                        std::vector<std::string> *out_tokens) const {
  CHECK_FAIL_RETURN_UNEXPECTED(start >= 0 && end > start && end <= input_token.size(), "Out of range");
  std::string subword = input_token.substr(start, end - start);
  if (start > 0) {
    subword = suffix_indicator_ + subword;
  }
  out_tokens->emplace_back(subword);
  return Status::OK();
}

Status WordpieceTokenizerOp::GetTokens(const std::string &input_token, const uint32_t &basic_start,
                                       std::vector<std::string> *out_tokens, std::vector<uint32_t> *offsets_start,
                                       std::vector<uint32_t> *offsets_limit) const {
  if (input_token.size() > max_bytes_per_token_) {
    offsets_start->push_back(basic_start);
    if (!unknown_token_.empty()) {
      offsets_limit->push_back(basic_start + unknown_token_.size());
      out_tokens->emplace_back(unknown_token_);
    } else {
      out_tokens->emplace_back(input_token);
      offsets_limit->push_back(basic_start + input_token.size());
    }
    return Status::OK();
  }
  RuneStrArray runes;
  if (!DecodeRunesInString(input_token.data(), input_token.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("WordpieceTokenizer: Decode utf8 string failed.");
  }
  int end = 0;
  for (int start = 0; start < input_token.size();) {
    bool found = false;
    RETURN_IF_NOT_OK(LookupWord(input_token, runes, start, &found, &end));
    if (found) {
      RETURN_IF_NOT_OK(AddSubword(input_token, start, end, out_tokens));
      offsets_start->push_back(static_cast<uint32_t>(basic_start + start));
      offsets_limit->push_back(static_cast<uint32_t>(basic_start + end));
      start = end;
    } else {
      return FoundNoToken(input_token, basic_start, out_tokens, offsets_start, offsets_limit);
    }
  }
  return Status::OK();
}

Status WordpieceTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (input[0]->Rank() > 1 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED(
      "WordpieceTokenizer: The input shape should be 1D scalar the input datatype should be string.");
  }
  dsize_t count = 0;
  std::vector<std::string> out_tokens;
  std::vector<uint32_t> offsets_start, offsets_limit;
  std::shared_ptr<Tensor> token_tensor, offsets_start_tensor, offsets_limit_tensor;
  for (auto iter = input[0]->begin<std::string_view>(); iter != input[0]->end<std::string_view>(); iter++) {
    uint32_t basic_start = 0;
    std::vector<std::string> temp_tokens;
    if (with_offsets_ && input.size() == 3) {
      RETURN_IF_NOT_OK(input[1]->GetItemAt<uint32_t>(&basic_start, {count, 0}));
    }
    RETURN_IF_NOT_OK(GetTokens(std::string(*iter), basic_start, &temp_tokens, &offsets_start, &offsets_limit));
    out_tokens.insert(out_tokens.end(), temp_tokens.begin(), temp_tokens.end());
    count++;
  }
  if (out_tokens.empty()) {
    out_tokens.emplace_back("");
    offsets_start.push_back(0);
    offsets_limit.push_back(0);
  }
  Tensor::CreateFromVector(out_tokens, &token_tensor);
  output->push_back(token_tensor);
  if (with_offsets_) {
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_start, &offsets_start_tensor));
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_limit, &offsets_limit_tensor));

    output->push_back(offsets_start_tensor);
    output->push_back(offsets_limit_tensor);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
