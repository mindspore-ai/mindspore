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
#include "minddata/dataset/text/kernels/jieba_tokenizer_op.h"

#include <vector>
#include <memory>
#include <string>
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {

JiebaTokenizerOp::JiebaTokenizerOp(const std::string &hmm_path, const std::string &dict_path, const JiebaMode &mode,
                                   const bool &with_offsets)
    : TokenizerOp(with_offsets), jieba_mode_(mode), hmm_model_path_(hmm_path), mp_dict_path_(dict_path) {
  jieba_parser_ = std::make_unique<cppjieba::Jieba>(mp_dict_path_, hmm_model_path_, "");
}

Status JiebaTokenizerOp::Tokenize(std::string_view sentence_v, std::vector<std::string> *words,
                                  std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) {
  std::string sentence{sentence_v};

  if (sentence == "") {
    (void)words->emplace_back("");
  } else {
    std::vector<cppjieba::Word> tmp;
    if (jieba_mode_ == JiebaMode::kMp) {
      std::unique_ptr<cppjieba::MPSegment> mp_seg = std::make_unique<cppjieba::MPSegment>(jieba_parser_->GetDictTrie());
      mp_seg->Cut(sentence, tmp, MAX_WORD_LENGTH);
    } else if (jieba_mode_ == JiebaMode::kHmm) {
      std::unique_ptr<cppjieba::HMMSegment> hmm_seg =
        std::make_unique<cppjieba::HMMSegment>(jieba_parser_->GetHMMModel());
      hmm_seg->Cut(sentence, tmp);
    } else {  // Mix
      std::unique_ptr<cppjieba::MixSegment> mix_seg =
        std::make_unique<cppjieba::MixSegment>(jieba_parser_->GetDictTrie(), jieba_parser_->GetHMMModel());
      mix_seg->Cut(sentence, tmp, true);
    }
    GetStringsFromWords(tmp, *words);
    for (auto item : tmp) {
      offsets_start->push_back(static_cast<uint32_t>(item.offset));
      offsets_limit->push_back(static_cast<uint32_t>(item.offset + item.word.length()));
    }
  }

  return Status::OK();
}

Status JiebaTokenizerOp::AddWord(const std::string &word, int freq) {
  RETURN_UNEXPECTED_IF_NULL(jieba_parser_);
  if (jieba_parser_->InsertUserWord(word, freq, "") == false) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "AddWord: add word failed.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
