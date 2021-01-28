/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_
#define DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_

#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace dataset {
namespace py = pybind11;

class BuildSentencePieceVocabOp : public PipelineOp {
 public:
  class Builder {
   public:
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method
    // @param uint32_t size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(uint32_t size) {
      builder_connector_size_ = size;
      return *this;
    }

    // Setter method
    // @param uint32_t size
    // @return Builder & reference to builder class object
    Builder &SetVocabSize(uint32_t size) {
      builder_vocab_size_ = size;
      return *this;
    }

    // Setter method
    // @param float character corverage - to determine the minimum symbols
    // @return Builder & reference to builder class object
    Builder &SetCharacterCoverage(float character_coverage) {
      builder_character_coverage_ = character_coverage;
      return *this;
    }

    // Setter method
    // @param SentencePieceModel model_type - model algorithm
    // @return Builder & reference to builder class object
    Builder &SetModelType(SentencePieceModel model_type) {
      builder_model_type_ = model_type;
      return *this;
    }

    // Setter method
    // @param std::unordered_map<std::string, std::string> params
    // @return Builder & reference to builder class object
    Builder &SetParams(std::unordered_map<std::string, std::string> params) {
      builder_params_ = params;
      return *this;
    }

    // Setter method
    // @param std::shared_ptr<SentencePieceVocab> vocab
    // @return Builder & reference to builder class object
    Builder &SetVocab(std::shared_ptr<SentencePieceVocab> vocab) {
      builder_vocab_ = vocab;
      return *this;
    }

    // set columns names
    // @param const std::vector<std::string> & col_names - name of columns to get words
    // @return Builder & reference to builder class object
    Builder &SetColumnNames(const std::vector<std::string> &col_names) {
      builder_col_names_ = col_names;
      return *this;
    }

    // The builder "build" method creates the final object.
    // @param std::shared_ptr<BuildVocabOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<BuildSentencePieceVocabOp> *op);

   private:
    uint32_t builder_connector_size_;
    uint32_t builder_vocab_size_;
    float builder_character_coverage_;
    SentencePieceModel builder_model_type_;
    std::unordered_map<std::string, std::string> builder_params_;
    std::vector<std::string> builder_col_names_;
    std::shared_ptr<SentencePieceVocab> builder_vocab_;
  };

 public:
  class DatasetSentenceIterator : public sentencepiece::SentenceIterator {
   public:
    explicit DatasetSentenceIterator(BuildSentencePieceVocabOp *s_p_vocab_ptr);
    ~DatasetSentenceIterator() {}

    bool done() const override;
    void Next() override;
    const std::string &value() const override { return value_; }
    sentencepiece::util::Status status() const override { return sentencepiece::util::Status(); }

   private:
    std::string value_;
    BuildSentencePieceVocabOp *s_p_vocab_ptr_;
  };

  BuildSentencePieceVocabOp(std::shared_ptr<SentencePieceVocab> vocab, std::vector<std::string> col_names,
                            int32_t vocab_size, float character_coverage, SentencePieceModel model_type,
                            const std::unordered_map<std::string, std::string> &params, int32_t op_conn_size);

  ~BuildSentencePieceVocabOp() = default;

  // the thread for sentence train
  Status SentenceThread();

  Status EofReceived(int32_t) override { return Status::OK(); }

  Status EoeReceived(int32_t) override { return Status::OK(); }

  Status operator()() override;

  // Getter
  // @return the number of workers
  int32_t num_producers() const override { return 1; }

  // Getter
  // @return the number of threads consuming from the previous Connector
  int32_t num_consumers() const override { return 1; }

  Status Reset() override { RETURN_STATUS_UNEXPECTED("Reset shouldn't be called in BuildSentencePieceVocabOp"); }

  std::string Name() const override { return kBuildSentencePieceVocabOp; }

  // build the input params for sentence api
  std::unordered_map<std::string, std::string> BuildParams();

  bool Done();
  void Next(std::string *sentence);

 private:
  bool read_done_;
  Status ret_status_;
  int32_t vocab_size_;
  float character_coverage_;
  SentencePieceModel model_type_;
  std::unordered_map<std::string, std::string> params_;
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::vector<std::string> col_names_;
  uint32_t col_id_;
  std::unique_ptr<ChildIterator> child_iterator_;     // child iterator for fetching TensorRows 1 by 1
  std::unique_ptr<Queue<TensorRow>> sentence_queue_;  // master thread assigns each worker TensorRow via this
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_
