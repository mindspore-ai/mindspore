/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#if defined(_WIN32) || defined(_WIN64)
#include <stdlib.h>
#endif
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
constexpr int32_t ShuffleOp::kShuffleStateInit;
constexpr int32_t ShuffleOp::kShuffleStateActive;
constexpr int32_t ShuffleOp::kShuffleStateDrain;

// Builder constructor. Creates the builder object.
ShuffleOp::Builder::Builder() : build_shuffle_size_(0), build_reshuffle_each_epoch_(true) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_op_connector_size_ = cfg->op_connector_size();
  build_rows_per_buffer_ = cfg->rows_per_buffer();
  build_shuffle_seed_ = GetSeed();
}

Status ShuffleOp::Builder::SanityCheck() const {
  if (build_shuffle_size_ < 2) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, shuffle buffer size must be greater than 1.");
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status ShuffleOp::Builder::Build(std::shared_ptr<ShuffleOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<ShuffleOp>(build_shuffle_size_, build_shuffle_seed_, build_op_connector_size_,
                                     build_reshuffle_each_epoch_, build_rows_per_buffer_);
  return Status::OK();
}

// Constructor of the ShuffleOp
ShuffleOp::ShuffleOp(int32_t shuffle_size, uint32_t shuffle_seed, int32_t op_connector_size, bool reset_every_epoch,
                     int32_t rows_per_buffer)
    : PipelineOp(op_connector_size),
      shuffle_size_(shuffle_size),
      shuffle_seed_(shuffle_seed),
      reshuffle_each_epoch_(reset_every_epoch),
      rng_(shuffle_seed),
      buffer_counter_(0),
      rows_per_buffer_(rows_per_buffer),
      shuffle_buffer_(std::make_unique<TensorTable>()),
      shuffle_last_row_idx_(0),
      shuffle_buffer_state_(kShuffleStateInit) {}

// Private function to re-init the shuffle op for another epoch.  Shuffle op calls this by
// itself rather than waiting for the reset driven from operators above it in the pipeline.
Status ShuffleOp::SelfReset() {
  MS_LOG(DEBUG) << "Shuffle operator performing a self-reset.";
  // If reshuffle_each_epoch is false, then we always use the same seed for every
  // epoch.
  // If reshuffle_each_epoch is true, then the first epoch uses the given seed,
  // and all subsequent epochs will then keep on using the rng_ without resetting it
  if (!reshuffle_each_epoch_) {
    rng_ = std::mt19937_64(shuffle_seed_);
  }

  shuffle_buffer_ = std::make_unique<TensorTable>();
  buffer_counter_ = 0;
  shuffle_last_row_idx_ = 0;
  shuffle_buffer_state_ = kShuffleStateInit;
  return Status::OK();
}

// A print method typically used for debugging
void ShuffleOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [shuffle size: " << shuffle_size_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nShuffle size: " << shuffle_size_ << "\nRows per buffer: " << rows_per_buffer_
        << "\nShuffle buffer state: " << shuffle_buffer_state_ << "\nShuffle seed: " << shuffle_seed_ << "\n\n";
  }
}

// Private function to add a new row to the shuffle buffer.
Status ShuffleOp::AddRowToShuffleBuffer(TensorRow new_shuffle_row) {
  // If the last slot of our shuffle buffer was not the full size of the shuffle buffer then we are
  // filling it during the initial fill codepath and thus growing it's size. In that case, we push
  // back the new row to grow our shuffle buffer size by 1.
  // If we are already at the full size, then we overwrite the last slot with our row (and the last
  // slot better be empty because it should already have been swapped out during the random row
  // selection that was done previously!)
  if (shuffle_last_row_idx_ < (shuffle_size_ - 1)) {
    shuffle_buffer_->push_back(std::move(new_shuffle_row));
    shuffle_last_row_idx_ = (shuffle_buffer_->size()) - 1;
  } else {
    if (!(*shuffle_buffer_)[shuffle_last_row_idx_].empty()) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                    "Last row of shuffle buffer should not be occupied!");
    }
    (*shuffle_buffer_)[shuffle_last_row_idx_] = std::move(new_shuffle_row);
  }
  return Status::OK();
}

// Class functor operator () override.
// All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work
Status ShuffleOp::operator()() {
  std::unique_ptr<TensorQTable> new_buffer_table;  // A tensor table to be used for output.

  // Synchronize with TaskManager once the thread is launched.
  TaskManager::FindMe()->Post();

  // Shuffle op does not have workers, and only consumes from child 0.
  // Create the child iterator to fetch our data from.
  int32_t worker_id = 0;
  int32_t child_idx = 0;
  child_iterator_ = std::make_unique<ChildIterator>(this, worker_id, child_idx);

  // Main operator loop
  while (true) {
    // Do an initial populate of the shuffle buffer
    RETURN_IF_NOT_OK(InitShuffleBuffer());

    // This is our main loop exit condition, when the iterator has no more data completely.
    if (child_iterator_->eof_handled()) {
      break;
    }

    // Next, enter into the main execution loop of the shuffle op.
    // When the tail index position of our shuffle buffer goes negative it means that we've
    // fully drained the data from the shuffle buffer and we're done.
    while (shuffle_last_row_idx_ >= 0) {
      // Step 1)
      // Create an output tensor table if one is not created yet.
      if (!new_buffer_table) {
        new_buffer_table = std::make_unique<TensorQTable>();
      }

      // Step 2)
      // Randomly select a slot from our shuffle buffer and copy that row into the output
      // tensor table. We remove the data from the shuffle buffer, leaving that slot
      // in the table as an empty vector
      int64_t random_slot = rng_() % (shuffle_last_row_idx_ + 1);
      new_buffer_table->push_back(std::move((*shuffle_buffer_)[random_slot]));

      // Step 3)
      // If the output tensor table is at the requested size, then create a buffer for it
      // and send this buffer on it's way up the pipeline. Special case is if this is the
      // last row then we also send it.
      if (new_buffer_table->size() == rows_per_buffer_ || shuffle_last_row_idx_ == 0) {
        auto new_buffer = std::make_unique<DataBuffer>(buffer_counter_, DataBuffer::kDeBFlagNone);
        new_buffer->set_tensor_table(std::move(new_buffer_table));
        buffer_counter_++;
        MS_LOG(DEBUG) << "Shuffle operator sending a buffer to output.";
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(new_buffer)));
      }

      // Step 4)
      // Take the last row from shuffle buffer, and swap it into the row position that was
      // just vacated.  This makes the shuffle buffer contiguous, with an empty slot at the
      // tail of the shuffle buffer.
      if (random_slot != shuffle_last_row_idx_) {
        (*shuffle_buffer_)[random_slot] = std::move((*shuffle_buffer_)[shuffle_last_row_idx_]);
      }

      // Step 5)
      // Refill the last slot of the shuffle buffer with the next row from input if we are in the
      // active state.
      // If we are in the draining state, we do not need to fetch another row to replace the one we
      // just drained.
      if (shuffle_buffer_state_ == kShuffleStateActive) {
        TensorRow new_row;
        RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

        if (!new_row.empty()) {
          RETURN_IF_NOT_OK(AddRowToShuffleBuffer(std::move(new_row)));
        } else {
          shuffle_buffer_state_ = kShuffleStateDrain;
        }
      }

      // If we are draining, reposition (decrement) our tail index in the shuffle buffer since we
      // just drained a row from it.
      if (shuffle_buffer_state_ == kShuffleStateDrain) {
        shuffle_last_row_idx_--;
      }
    }

    // Since we overloaded eoeReceived function, we are responsible to flow the EOE up the
    // pipeline manually now that we are done draining the shuffle buffer
    MS_LOG(DEBUG) << "Shuffle operator sending EOE.";
    auto eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));

    // Do not wait for any reset to be flown down from operators above us.
    // Instead, manually update ourselves and then go reloop to start fetching from child operator
    // right away.  Any Reset() from the parent will still perform common reset actions.
    RETURN_IF_NOT_OK(this->SelfReset());
  }
  return Status::OK();
}

// Private function populate the shuffle buffer initially by fetching from the child output
// connector until the shuffle buffer is full (or there is no more data coming).
Status ShuffleOp::InitShuffleBuffer() {
  MS_LOG(DEBUG) << "Shuffle operator initializing the shuffle buffer.";

  // The first phase of this operator is to read incoming buffers and then drain those
  // rows from the buffers, putting them into our own local table of tensors (the shuffle
  // buffer).
  // This shuffle buffer initialization phase stops when we've either filled up the
  // shuffle buffer to it's max size, or the dataset below us is not providing any more
  // rows.
  if (shuffle_buffer_state_ != kShuffleStateInit) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid shuffle buffer state (SHUFFLE_STATE_INIT expected)");
  }

  // Before we drop into the fetching loop, call the fetch once for the first time
  // to fill the first row and grab the first buffer.
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

  if (child_iterator_->eof_handled()) {
    MS_LOG(DEBUG) << "Shuffle operator init picked up EOF. No more epochs.";
    return Status::OK();
  }

  if (new_row.empty()) {
    RETURN_STATUS_UNEXPECTED("Unable to fetch a single row for shuffle buffer.");
  }

  // Now fill the rest of the shuffle buffer until we are unable to get the next row or we reached
  // the desired shuffle buffer size.
  while (!new_row.empty() && shuffle_buffer_->size() < static_cast<size_t>(shuffle_size_ - 1)) {
    // Add the previously fetched row
    RETURN_IF_NOT_OK(AddRowToShuffleBuffer(std::move(new_row)));

    // Fetch the next row
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }

  // If we quit the loop due to being at the shuffle size, still need to add the last row here.
  if (!new_row.empty()) {
    RETURN_IF_NOT_OK(AddRowToShuffleBuffer(std::move(new_row)));
    shuffle_buffer_state_ = kShuffleStateActive;  // Transition to the active state
  } else {
    // If init phase doesn't have more rows, then skip the active state and jump straight to the
    // shuffle buffer draining state
    shuffle_buffer_state_ = kShuffleStateDrain;
  }

  MS_LOG(DEBUG) << "Shuffle operator finished initializing the shuffle buffer.";
  return Status::OK();
}

Status ShuffleOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
