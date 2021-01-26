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

const bool JiebaTokenizerOp::kDefWithOffsets = false;

JiebaTokenizerOp::JiebaTokenizerOp(const std::string &hmm_path, const std::string &dict_path, const JiebaMode &mode,
                                   const bool &with_offsets)
    : jieba_mode_(mode), hmm_model_path_(hmm_path), mp_dict_path_(dict_path), with_offsets_(with_offsets) {
  jieba_parser_ = std::make_unique<cppjieba::Jieba>(mp_dict_path_, hmm_model_path_, "");
}

Status JiebaTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "JiebaTokenizer: input only support one column data.");
  RETURN_UNEXPECTED_IF_NULL(jieba_parser_);

  if (input[0]->Rank() != 0 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("JiebaTokenizer: the input should be scalar with string datatype.");
  }

  std::string_view sentence_v;
  RETURN_IF_NOT_OK(input[0]->GetItemAt(&sentence_v, {}));
  std::string sentence{sentence_v};
  std::vector<std::string> words;
  std::vector<uint32_t> offsets_start, offsets_limit;
  std::shared_ptr<Tensor> token_tensor, offsets_start_tensor, offsets_limit_tensor;
  if (sentence == "") {
    words.push_back("");
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
    GetStringsFromWords(tmp, words);
    for (auto item : tmp) {
      offsets_start.push_back(static_cast<uint32_t>(item.offset));
      offsets_limit.push_back(static_cast<uint32_t>(item.offset + item.word.length()));
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(words, &token_tensor));
  output->push_back(token_tensor);
  if (with_offsets_) {
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_start, &offsets_start_tensor));
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_limit, &offsets_limit_tensor));

    output->push_back(offsets_start_tensor);
    output->push_back(offsets_limit_tensor);
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
