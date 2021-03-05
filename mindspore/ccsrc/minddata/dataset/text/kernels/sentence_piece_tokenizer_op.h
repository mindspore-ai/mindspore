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

#ifndef DATASET_SENTENCE_PIECE_TOKENIZER_OP_H
#define DATASET_SENTENCE_PIECE_TOKENIZER_OP_H

#include <sentencepiece_processor.h>

#include <string>
#include <iostream>
#include <memory>

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"

namespace mindspore {
namespace dataset {

class SentencePieceTokenizerOp : public TensorOp {
 public:
  SentencePieceTokenizerOp(const std::shared_ptr<SentencePieceVocab> vocab, SPieceTokenizerLoadType load_type,
                           const SPieceTokenizerOutType out_type);

  SentencePieceTokenizerOp(const std::string &model_path, const std::string &model_filename,
                           const SPieceTokenizerLoadType load_type, const SPieceTokenizerOutType out_type);

  ~SentencePieceTokenizerOp() override = default;

  Status GetModelRealPath(const std::string &model_path, const std::string &filename);

  void Print(std::ostream &out) const override {
    out << Name() << " out_type = " << out_type_ << " load_type = " << load_type_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSentencepieceTokenizerOp; }

 protected:
  SPieceTokenizerOutType out_type_;
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::string file_path_;
  SPieceTokenizerLoadType load_type_;
  sentencepiece::SentencePieceProcessor processor_;
  Status model_status_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_SENTENCE_PIECE_TOKENIZER_OP_H
