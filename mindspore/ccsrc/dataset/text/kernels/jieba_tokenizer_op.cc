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
#include "dataset/text/kernels/jieba_tokenizer_op.h"

#include <vector>
#include <memory>
#include <string>
#include "dataset/util/path.h"

namespace mindspore {
namespace dataset {

JiebaTokenizerOp::JiebaTokenizerOp(const std::string &hmm_path, const std::string &dict_path, JiebaMode mode)
    : jieba_mode_(mode), hmm_model_path_(hmm_path), mp_dict_path_(dict_path) {
  jieba_parser_ = std::make_unique<cppjieba::Jieba>(mp_dict_path_, hmm_model_path_, "");
}

Status JiebaTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_UNEXPECTED_IF_NULL(jieba_parser_);

  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("the input tensor should be scalar string tensor");
  }

  std::string_view sentence_v;
  RETURN_IF_NOT_OK(input->GetItemAt(&sentence_v, {}));
  std::string sentence{sentence_v};
  std::vector<std::string> words;
  if (sentence == "") {
    words.push_back("");
  } else {
    if (jieba_mode_ == JiebaMode::kMp) {
      jieba_parser_->CutSmall(sentence, words, MAX_WORD_LENGTH);
    } else if (jieba_mode_ == JiebaMode::kHmm) {
      jieba_parser_->CutHMM(sentence, words);
    } else {  // Mix
      jieba_parser_->Cut(sentence, words, true);
    }
  }
  *output = std::make_shared<Tensor>(words, TensorShape({(dsize_t)words.size()}));
  return Status::OK();
}

Status JiebaTokenizerOp::AddWord(const std::string &word, int freq) {
  RETURN_UNEXPECTED_IF_NULL(jieba_parser_);
  if (jieba_parser_->InsertUserWord(word, freq, "") == false) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "add word error");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
