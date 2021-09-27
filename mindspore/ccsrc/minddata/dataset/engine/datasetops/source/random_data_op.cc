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

#include "minddata/dataset/engine/datasetops/source/random_data_op.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

namespace mindspore {
namespace dataset {
// Constructor for RandomDataOp
RandomDataOp::RandomDataOp(int32_t num_workers, int32_t op_connector_size, int64_t total_rows,
                           std::unique_ptr<DataSchema> data_schema)
    : ParallelOp(num_workers, op_connector_size),
      total_rows_(total_rows),
      epoch_rows_sent_(0),
      guys_in_(0),
      guys_out_(num_workers_),
      eoe_worker_id_(0),
      data_schema_(std::move(data_schema)) {
  rand_gen_.seed(GetSeed());  // seed the random generator
  // If total rows was not given, then randomly pick a number
  if (total_rows_ == 0) {
    total_rows_ = GenRandomInt(1, kMaxTotalRows);
  }
  // If the user did not provide a schema, then we will ask the op to generate a pseudo-random schema.
  // See details of generateSchema function to learn what type of schema it will create.
  if (data_schema_ == nullptr) {
    GenerateSchema();
  }
  // Everyone is already out from the sync area.
  all_out_.Set();
}

// A print method typically used for debugging
void RandomDataOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [total rows: " << total_rows_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nTotal_rows: " << total_rows_ << " \nSchema:\n" << *data_schema_ << "\n\n";
  }
}

// Helper function to produce a default/random schema if one didn't exist
void RandomDataOp::GenerateSchema() {
  const int32_t type_offset = 2;
  // To randomly create a schema, we need to choose:
  // a) how many columns
  // b) the type of each column
  // c) the shape of each column (number of dimensions i.e. rank)
  // d) the shape of each column (dimension values)
  data_schema_ = std::make_unique<DataSchema>();
  std::unique_ptr<TensorShape> new_shape;
  std::unique_ptr<ColDescriptor> new_col;

  // Loop over the number of chosen columns
  int32_t numColumns = GenRandomInt(1, kMaxNumColumns);
  for (int32_t i = 0; i < numColumns; i++) {
    // For each column:
    // - choose a datatype
    // - generate a shape that randomly chooses the number of dimensions and the dimension values.
    DataType::Type newType = static_cast<DataType::Type>(GenRandomInt(1, DataType::NUM_OF_TYPES - type_offset));
    int32_t rank = GenRandomInt(1, kMaxRank);
    std::vector<dsize_t> dims;
    for (int32_t d = 0; d < rank; d++) {
      // 0 is not a valid dimension value.  however, we can support "*" or unknown, so map the random
      // 0 value to the unknown attribute if 0 is chosen
      dsize_t dim_value = static_cast<dsize_t>(GenRandomInt(0, kMaxDimValue));
      if (dim_value == 0) {
        dim_value = TensorShape::kDimUnknown;
      }
      dims.push_back(dim_value);
    }
    new_shape = std::make_unique<TensorShape>(dims);

    // Create the column descriptor
    std::string col_name = "c" + std::to_string(i);
    new_col =
      std::make_unique<ColDescriptor>(col_name, DataType(newType), TensorImpl::kFlexible, rank, new_shape.get());

    Status rc = data_schema_->AddColumn(*new_col);
    if (rc.IsError()) MS_LOG(ERROR) << "Failed to generate a schema. Message:" << rc;
  }
}

// Class functor operator () override.
// All DatasetOps operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work.
Status RandomDataOp::operator()() {
  CHECK_FAIL_RETURN_UNEXPECTED(total_rows_ >= num_workers_,
                               "RandomDataOp expects total_rows < num_workers. Try adjust num_workers, total_row=" +
                                 std::to_string(total_rows_) + ", num_workers=" + std::to_string(num_workers_) + " .");

  // If the amount of workers we have exceeds the number of rows to produce, then we'll have
  // idle workers doing nothing.  In that case, let's throttle the worker count.
  if (num_workers_ > total_rows_) {
    MS_LOG(INFO) << "RandomDataOp throttling worker count from " << num_workers_ << "to " << total_rows_;
    num_workers_ = total_rows_;
    num_producers_ = num_workers_;
    guys_out_ = num_workers_;
    // The output connector was already created with a different worker count.  We have to drop and recreate
    // that connector.
    DatasetOp::CreateConnector(num_producers_, num_workers_);
  }

  if (num_workers_ == 0) {
    RETURN_STATUS_UNEXPECTED("Invalid data, num_workers_ is zero.");
  }
  // Assign the number of rows to each worker in a round robin fashion.
  worker_max_rows_.reserve(num_workers_);
  worker_rows_packed_.reserve(num_workers_);
  // init the counts to zero to start.
  for (int32_t w = 0; w < num_workers_; w++) {
    worker_max_rows_.push_back(0);
    worker_rows_packed_.push_back(0);
  }
  // then assign round robin row counts
  int32_t currentWorker = 0;
  for (int64_t r = 0; r < total_rows_; r++) {
    worker_max_rows_[currentWorker]++;
    currentWorker = (currentWorker + 1) % num_workers_;
  }

  // Next, compute the total rows count.  This stat is needed during reset logic
  for (int32_t w = 0; w < num_workers_; w++) {
    epoch_rows_sent_ += worker_max_rows_[w];
  }

  // For the connector to work, we need to target the correct worker channel for the eoe.
  // This will initialize it for the first one.  reset() handles for the rest of the epochs.
  eoe_worker_id_ = epoch_rows_sent_ % num_workers_;
  epoch_rows_sent_++;  // Add the eoe row to the count for subsequent epochs

  // RandomDataOp doesn't need the master thread to stay around.  Kick off the workers and then master exits.
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&RandomDataOp::WorkerEntry, this, std::placeholders::_1), "", id()));

  // required task group setup after launching workers
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(epoch_sync_wait_post_.Register(tree_->AllTasks()));

  return Status::OK();
}

// Performs a synchronization between workers at the end of an epoch
Status RandomDataOp::EpochSync(int32_t worker_id, bool *quitting) {
  MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " syncing at end of epoch";

  // Sync on the guys_in counter
  // We have to wait the last guy is out.
  RETURN_IF_NOT_OK(all_out_.Wait());
  // If we are not in a repeat loop, or that was the last repeat already, then setup our exit
  // condition from the master loop.
  if (IsLastIteration()) {
    *quitting = true;
  }

  auto prev = guys_in_.fetch_add(1);
  bool last_guy_in = (prev + 1) == num_workers_;
  // If we are the last worker to hit this sync point, we have some extra tasks
  if (last_guy_in) {
    MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " is the last one to sync. eoe sent as worker "
                 << eoe_worker_id_;
    UpdateRepeatAndEpochCounter();
    // Prepare for sync
    all_out_.Clear();
    // Always flow eoe at the end
    RETURN_IF_NOT_OK(out_connector_->SendEOE(eoe_worker_id_));
    // If we're done then also flow the eof
    if (*quitting) {
      // The eof needs to be sent from the next sender in the round robin, so +1
      int32_t eof_worker_id = (eoe_worker_id_ + 1) % num_workers_;
      MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " has no more epochs.  sending eof as worker "
                   << eof_worker_id;
      RETURN_IF_NOT_OK(out_connector_->SendEOF(eof_worker_id));
    }
  }

  if (!(*quitting)) {
    MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " entering sync wait.";
    if (last_guy_in) {
      // If we are the last worker, do reset to wake other workers up
      RETURN_IF_NOT_OK(Reset());
    } else {
      // If we are not the last worker, wait for the reset
      RETURN_IF_NOT_OK(epoch_sync_wait_post_.Wait());
    }
    prev = guys_out_.fetch_add(1);
    bool last_guy_out = (prev + 1) == num_workers_;
    // Last guy out will clear the wait post and set the row counts
    if (last_guy_out) {
      MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " last guy out clearing wait post.";
      epoch_sync_wait_post_.Clear();
      guys_in_ = 0;
      all_out_.Set();
    }
  }

  MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " epoch sync complete.";
  return Status::OK();
}

// The entry point code for when workers are launched
Status RandomDataOp::WorkerEntry(int32_t worker_id) {
  MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " entry";

  // handshake with the master first to tell it we're alive
  TaskManager::FindMe()->Post();

  bool quitting = false;
  std::unique_ptr<TensorQTable> new_tensor_table = nullptr;

  // Loop until the quitting variable gets set to true
  do {
    // If we have not yet reached the row count for this worker then produce another record
    if (worker_rows_packed_[worker_id] < worker_max_rows_[worker_id]) {
      TensorRow new_row;

      // Start a new tensor table if needed
      if (new_tensor_table == nullptr) {
        new_tensor_table = std::make_unique<TensorQTable>();
      }

      // Create the data for the row
      RETURN_IF_NOT_OK(CreateRandomRow(worker_id, &new_row));

      // Add the row to our table
      worker_rows_packed_[worker_id]++;

      // Send new_row out
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row), worker_id));
    } else {
      // Now, let's enter the epoch sync
      RETURN_IF_NOT_OK(EpochSync(worker_id, &quitting));
    }
  } while (!quitting);

  MS_LOG(INFO) << "RandomDataOp worker " << worker_id << " is now quitting.";

  return Status::OK();
}

// A helper function to create random data for the row
Status RandomDataOp::CreateRandomRow(int32_t worker_id, TensorRow *new_row) {
  if (new_row == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "[Internal ERROR] Missing tensor row output.");
  }

  // Create a tensor for each column, then add the tensor to the row
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    const ColDescriptor current_col = data_schema_->Column(i);
    std::vector<dsize_t> current_shape = current_col.Shape().AsVector();
    std::unique_ptr<TensorShape> new_shape = nullptr;
    std::unique_ptr<unsigned char[]> buf = nullptr;
    std::shared_ptr<Tensor> new_tensor = nullptr;

    // We need to resolve the shape to fill in any unknown dimensions with random
    // values, then use that as our shape for this tensor.
    for (int j = 0; j < current_shape.size(); ++j) {
      if (current_shape[j] == TensorShape::kDimUnknown) {
        current_shape[j] = static_cast<dsize_t>(GenRandomInt(1, kMaxDimValue));
      }
    }

    new_shape = std::make_unique<TensorShape>(current_shape);
    int64_t size_in_bytes = new_shape->NumOfElements() * current_col.Type().SizeInBytes();

    // Generate a random byte of data.  This may cause some funny data for things like doubles,floats, bools
    // however the random data op is not too concerned about the physical data itself.
    std::uniform_int_distribution<uint8_t> uniDist(0, UINT8_MAX);
    uint8_t random_byte = uniDist(rand_gen_);

    // Now, create a chunk of memory for the entire tensor and copy this byte in repeatedly.
    buf = std::make_unique<unsigned char[]>(size_in_bytes);
    int ret_code = memset_s(buf.get(), size_in_bytes, random_byte, size_in_bytes);
    if (ret_code != 0) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Failed to set random bytes for a tensor.");
    }

    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(*new_shape, current_col.Type(), buf.get(), &new_tensor));

    // Add this tensor to the tensor row for output
    (*new_row).push_back(std::move(new_tensor));
  }
  return Status::OK();
}

// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status RandomDataOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";

  // Ensure all guys are in the waitpost
  if (guys_in_ != num_workers_) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Issuing a reset, but some workers are missing from epochSync!");
  }

  // reset the row counters for all workers
  for (int32_t w = 0; w < num_workers_; w++) {
    worker_rows_packed_[w] = 0;
    worker_max_rows_[w] = 0;
  }

  // Re-assign round robin row counts, starting from the worker after the one that gave
  // the eoe last time
  int32_t currentWorker = (eoe_worker_id_ + 1) % num_workers_;
  for (int64_t r = 0; r < total_rows_; r++) {
    worker_max_rows_[currentWorker]++;
    currentWorker = (currentWorker + 1) % num_workers_;
  }

  // Compute which worker should get the eoe for the next epoch
  eoe_worker_id_ = ((epoch_rows_sent_ % num_workers_) + eoe_worker_id_) % num_workers_;

  // Wake up the workers to get them going again in a new epoch
  guys_out_ = 0;
  epoch_sync_wait_post_.Set();

  return Status::OK();
}

Status RandomDataOp::ComputeColMap() {
  // Extract the column name mapping from the schema and save it in the class.
  if (column_name_id_map_.empty()) {
    RETURN_IF_NOT_OK(data_schema_->GetColumnNameMap(&(column_name_id_map_)));
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
