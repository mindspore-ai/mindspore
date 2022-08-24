/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TEXT_JIEBA_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TEXT_JIEBA_OP_H_

#include <string>
#include <vector>
#include <memory>
#include "cppjieba/Jieba.hpp"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/tokenizer_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class JiebaTokenizerOp : public TokenizerOp {
 public:
  // default constant for Jieba MPSegment algorithm.
  static constexpr size_t MAX_WORD_LENGTH = 512;
  // Constructor for JiebaTokenizerOp.
  // @param hmm_path HMM model file.
  // @param mp_path MP model file.
  // @mode tokenization mode [Default "MIX"], "MP" model will tokenize with MPSegment algorithm, "HMM" mode will
  // tokenize with Hiddel Markov Model Segment algorithm, "MIx" model will tokenize with a mix of MPSegment and
  // HMMSegment algorithm.
  // @with_offsets user set this value to choose whether output offset tensor.
  JiebaTokenizerOp(const std::string &hmm_path, const std::string &mp_path, const JiebaMode &mode = JiebaMode::kMix,
                   const bool &with_offsets = kDefWithOffsets);
  ~JiebaTokenizerOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << static_cast<int>(jieba_mode_) << "hmm_model_path_ " << hmm_model_path_ << "mp_dict_path_"
        << mp_dict_path_;
  }

  Status Tokenize(std::string_view sentence_v, std::vector<std::string> *words, std::vector<uint32_t> *offsets_start,
                  std::vector<uint32_t> *offsets_limit) override;

  // @word the word to be added to the JiebaTokenizer.
  // @freq [Default 0] the frequency fo the word to be added.
  // @tag [Default ""] the tag of the word to be added.
  Status AddWord(const std::string &word, int freq = 0);

  std::string Name() const override { return kJiebaTokenizerOp; }

 protected:
  JiebaMode jieba_mode_;
  std::string hmm_model_path_;
  std::string mp_dict_path_;
  std::unique_ptr<cppjieba::Jieba> jieba_parser_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TEXT_JIEBA_OP_H_
