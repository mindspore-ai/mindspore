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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/text/vocab.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class BuildVocabOp : public ParallelOp {
 public:
  class Builder {
   public:
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method
    // @param int32_t size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t size) {
      builder_connector_size_ = size;
      return *this;
    }

    // Setter method
    // @param int32_t num_workers
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method
    // @param int64_t top_k
    // @return Builder setter method returns reference to the builder.
    Builder &SetTopK(int64_t top_k) {
      builder_top_k_ = top_k;
      return *this;
    }

    // Setter method
    // @param int64_t min_freq
    // @return Builder setter method returns reference to the builder.
    Builder &SetMinFreq(int64_t min_freq) {
      builder_min_freq_ = min_freq;
      return *this;
    }

    // Setter method
    // @param int64_t max_freq
    // @return Builder setter method returns reference to the builder.
    Builder &SetMaxFreq(int64_t max_freq) {
      builder_max_freq_ = max_freq;
      return *this;
    }

    // set columns names
    // @param const std::vector<std::string> & col_names - name of columns to get words
    // @return Builder & reference to builder class object
    Builder &SetColumnNames(const std::vector<std::string> &col_names) {
      builder_col_names_ = col_names;
      return *this;
    }

    // set special tokens
    // @param const std::vector<std::string> & col_names - name of columns to get words
    // @return Builder & reference to builder class object
    Builder &SetSpecialTokens(const std::vector<std::string> &tokens) {
      builder_speical_tokens_ = tokens;
      return *this;
    }

    // set vocab object
    Builder &SetVocab(std::shared_ptr<Vocab> vocab) {
      builder_vocab_ = vocab;
      return *this;
    }

    // set special tokens first (or last)
    Builder &SetSpecialFirst(bool prepend) {
      builder_special_first_ = prepend;
      return *this;
    }

    // The builder "build" method creates the final object.
    // @param std::shared_ptr<BuildVocabOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<BuildVocabOp> *op);

   private:
    int32_t builder_num_workers_;
    int32_t builder_connector_size_;
    int64_t builder_min_freq_;
    int64_t builder_max_freq_;
    bool builder_special_first_;
    std::vector<std::string> builder_col_names_;
    std::vector<std::string> builder_speical_tokens_;
    std::shared_ptr<Vocab> builder_vocab_;
    int64_t builder_top_k_;
  };

  BuildVocabOp(std::shared_ptr<Vocab> vocab, std::vector<std::string> col_names, std::pair<int64_t, int64_t> freq_range,
               int64_t top_k, const std::vector<std::string> &tokens, bool prepend, int32_t num_workers,
               int32_t op_connector_size);

  ~BuildVocabOp() = default;

  /// \brief A print method typically used for debugging
  /// \param[out] out The output stream to write output to
  /// \param[in] show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kBuildVocabOp; }

  /// \briefStream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param[out] out Reference to the output stream being overloaded
  /// \param[in] vop - reference to the BuildVocabOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const BuildVocabOp &vop) {
    vop.Print(out, false);
    return out;
  }

  Status WorkerEntry(int32_t worker_id) override;

  // collect the work product from each worker
  Status CollectorThread();

  Status EofReceived(int32_t) override { return Status::OK(); }

  Status EoeReceived(int32_t) override { return Status::OK(); }

  Status operator()() override;

  // Getter
  // @return the number of workers
  int32_t num_producers() const override { return 1; }

  // Getter
  // @return the number of threads consuming from the previous Connector
  int32_t num_consumers() const override { return 1; }

  Status Reset() override { RETURN_STATUS_UNEXPECTED("Reset shouldn't be called in BuildVocabOp"); }

 private:
  const int32_t interval_;
  bool special_first_;
  std::shared_ptr<Vocab> vocab_;
  std::vector<std::string> col_names_;
  std::vector<int32_t> col_ids_;
  std::vector<std::string> special_tokens_;
  // pair = {min_f, max_f}
  // make sure that 0<= min_f < max_f <= int32_max in the builder
  std::pair<int64_t, int64_t> freq_range_;

  int64_t top_k_;                                        // every thing means top_k_ == int32_max
  std::unique_ptr<ChildIterator> child_iterator_;        // child iterator for fetching TensorRows 1 by 1
  std::unique_ptr<Queue<TensorRow>> distributor_queue_;  // master thread assigns each worker TensorRow via this
  std::unique_ptr<Queue<std::unique_ptr<std::unordered_map<std::string, int64_t>>>> collector_queue_;
  std::unordered_map<std::string, int64_t> word_cnt_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_
