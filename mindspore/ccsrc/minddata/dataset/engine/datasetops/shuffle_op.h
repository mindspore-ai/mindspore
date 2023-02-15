/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SHUFFLE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SHUFFLE_OP_H_

#include <map>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class ShuffleOp : public PipelineOp {
  // Shuffle buffer state flags
  //
  // Shuffle buffer is in a state of being initialized
  static constexpr int32_t kShuffleStateInit = 0;

  // Shuffle buffer is in a state of being actively drained from, but refilling as well
  static constexpr int32_t kShuffleStateActive = 1;

  // Shuffle buffer is in a state of being drained
  static constexpr int32_t kShuffleStateDrain = 2;

 public:
  // Constructor of the ShuffleOp
  // @note The builder class should be used to call it
  // @param shuffle_size - The size for the shuffle buffer
  // @param shuffle_seed - The seed to use for random number generation
  // @param op_connector_size - The output connector queue size
  ShuffleOp(int32_t shuffle_size, uint32_t shuffle_seed, int32_t op_connector_size, bool reset_every_epoch);

  // Destructor
  ~ShuffleOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param so - reference to the ShuffleOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ShuffleOp &so) {
    so.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Base-class override for special eoe handler.
  // ShuffleOp must override this because it shall not perform default handling of eoe. Instead
  // the ShuffleOp needs to manage actions related to the end of the epoch itself.
  // @return Status The status code returned
  Status EoeReceived(int32_t worker_id) override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kShuffleOp; }

  // \brief During tree prepare phase, operators may have specific post-operations to perform depending on
  //     their role.
  // \notes Derived versions of this function should always call their superclass version first
  //     before providing their own implementations.
  // @return Status The status code returned
  Status PrepareOperator() override;

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  // Private function to add a new row to the shuffle buffer.
  // @return Status The status code returned
  Status AddRowToShuffleBuffer(TensorRow new_shuffle_row);

  /// \brief Private function to populate the shuffle buffer initially by fetching from the child output
  ///     connector until the shuffle buffer is full (or there is no more data coming).
  /// \param is_pull_mode - flag to indicate if pull mode is on
  /// \return Status The status code returned
  Status InitShuffleBuffer(bool is_pull_mode);

  /// \brief Gets one row out of the shuffle buffer and fills the vacant row with the last row in the buffer. This
  ///     implemented function is for both pull mode and non-pull mode. If it's in non-pull mode, fetch data from the
  ///     internal child iterator. Otherwise, fetch by calling GetNextRowPullMode() of its child node.
  /// \param row[out] - Fetched TensorRow
  /// \param is_pull_mode - flag to indicate if pull mode is on
  /// \return Status The status code returned
  Status GetShuffledRowImpl(TensorRow *row, bool is_pull_mode);

  // Private function to re-init the shuffle op for another epoch.  Shuffle op calls this by
  // itself rather than waiting for the reset driven from operators above it in the pipeline.
  // @return Status The status code returned
  Status SelfReset();

  int32_t shuffle_size_;  // User config for the size of the shuffle buffer (number of rows)
  uint32_t shuffle_seed_;
  bool reshuffle_each_epoch_;
  // rng_ is seeded initially with shuffle_seed_. mt19937 is used for its large period.
  // specifically mt19937_64 is used to generate larger random numbers to reduce bias when
  // modding to fit within our desired range. we dont use a distribution
  // (ie uniform_int_distribution) because we will need to create up to |dataset| instances
  // of the distribution object in the common case of a perfect shuffle
  std::mt19937_64 rng_;
  // A single (potentially large) buffer of tensor rows for performing shuffling.
  std::unique_ptr<TensorTable> shuffle_buffer_;
  int32_t shuffle_last_row_idx_;  // Internal tracking of the last slot of our shuffle buffer
  int32_t shuffle_buffer_state_;  // State tracking for the shuffle buffer phases of work

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.
  bool eof_received_{false};                       // flag to indicate if eof is reached in pull mode.
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SHUFFLE_OP_H_
