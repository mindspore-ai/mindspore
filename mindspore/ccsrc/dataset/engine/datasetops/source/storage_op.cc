/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#define MAX_INTEGER_UINT32 4294967295
#define MAX_INTEGER_INT32 2147483647

#include "dataset/engine/datasetops/source/storage_client.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/constants.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/data_schema.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/util/queue.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "dataset/util/task_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
// Builder constructor.  Creates the builder object.
StorageOp::Builder::Builder()
    : build_dataset_files_dir_(""),
      build_schema_file_(""),
      build_num_rows_(0),
      build_data_distribution_file_(""),
      build_batch_size_(1),
      build_drop_remainder_(false) {
  // Some arguments to the StorageOp constructor have a default argument that is taken
  // from the client config.
  // The user may choose to change these values for the construction of the StorageOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_rows_per_buffer_ = cfg->rows_per_buffer();
  build_worker_connector_size_ = cfg->worker_connector_size();
  build_num_workers_ = cfg->num_parallel_workers();
  build_op_connector_size_ = cfg->op_connector_size();
}

// The builder "build" method creates the final object.
Status StorageOp::Builder::Build(std::shared_ptr<StorageOp> *ptr) {
  // There are 2 "flavours" of construction for a StorageOp:
  //
  // 1) Does a handshake with the dataset to identify row ranges and to identify
  //    the schema (internally the handshake does lookup against a json file in the dataset)
  //
  // 2) The user manually creates a schema and defines the row ranges, so there is no real
  //    dataset handshake.
  //
  // The decision about which style is called will depend on if the user supplied the
  // schema and row range fields.

  const std::string dataset_schema_file("datasetSchema.json");
  if (build_schema_ != nullptr && build_num_rows_ == 0) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a StorageOp with a given schema, but the number of rows not specified!");
  }
  if (build_schema_ == nullptr && build_num_rows_ != 0) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a StorageOp with a given number of rows but schema not specified!");
  }
  if (build_dataset_files_dir_.empty() && build_dataset_file_list_.empty()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a StorageOp that has not provided the location of the data files.");
  }
  if (!build_dataset_files_dir_.empty() && !build_dataset_file_list_.empty()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a StorageOp that has provided conflicting location of the data files.");
  }

  std::shared_ptr<StorageOp> new_storage_op = std::make_shared<StorageOp>(
    build_num_workers_, build_worker_connector_size_, build_rows_per_buffer_, build_op_connector_size_,
    build_columns_to_load_, build_data_distribution_file_, build_batch_size_, build_drop_remainder_);

  // If there is no schema or number of rows given, then we go with construction method 1
  // where we need to handshake with storage client to find out what the schema (and
  // number of rows) are based on schema file.
  if (build_schema_ == nullptr && build_num_rows_ == 0) {
    if (!build_dataset_files_dir_.empty()) {
      // We have a dataset files dir, but do not have a schema file.
      // Set the default schema file to be inside the same path as the dataset files dir.
      if (build_schema_file_.empty()) {
        build_schema_file_ = build_dataset_files_dir_ + "/" + dataset_schema_file;
      }
      RETURN_IF_NOT_OK(new_storage_op->InitOp(build_dataset_files_dir_, build_schema_file_, build_labels_file_name_,
                                              build_dataset_usage_));
    } else {
      // dataset is provided by list of files not dir_path
      RETURN_IF_NOT_OK(new_storage_op->InitOp(build_dataset_file_list_, build_schema_file_));
    }
  } else {
    // else, the user gave us a schema and a row range, go with construction method 2, where we use
    // the user-provided schema, but we still need to identify our data files.
    RETURN_IF_NOT_OK(new_storage_op->InitOp(build_num_rows_, build_dataset_files_dir_, std::move(build_schema_),
                                            build_labels_file_name_, build_dataset_usage_));
  }

  // Call the actual workhorse of the constructor
  RETURN_IF_NOT_OK(new_storage_op->init());
  *ptr = std::move(new_storage_op);
  return Status::OK();
}

StorageOp::StorageOp(int32_t num_workers, int32_t worker_connector_size, int32_t rows_per_buffer,
                     int32_t op_connector_size, std::vector<std::string> columns_to_load,
                     std::string data_distribution_file, int32_t batch_size, bool drop_remainder)
    : ParallelOp(num_workers, op_connector_size),
      worker_conn_size_(worker_connector_size),
      rows_per_buffer_(rows_per_buffer),
      num_rows_(0),
      buffers_fetched_(0),
      columns_to_load_(columns_to_load),
      data_distribution_file_(data_distribution_file),
      device_num_(1),
      device_id_(0),
      shard_config_("ALL"),
      seed_(0),
      shuffle_config_(false),
      num_classes_(0),
      batch_size_(batch_size),
      drop_remainder_(drop_remainder) {}

// Init of the StorageOp.  This is 1 of 3 init.
// This version of the init does not take the schema in it's arguments. It must perform an
// internal handshake with the dataset to produce the schema.
Status StorageOp::InitOp(const std::string &dataset_files_dir, const std::string &schema_file,
                         const std::string &labels_file_name, const std::string &dataset_usage) {
  dataset_files_dir_ = dataset_files_dir;
  schema_file_ = schema_file;
  labels_file_name_ = labels_file_name;
  dataset_usage_ = dataset_usage;

  // Storage ops require the internal master/worker connector.  create it here
  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_conn_size_));

  // Get parameter for distribution.
  RETURN_IF_NOT_OK(LoadParallelConfig());

  // Create the storage client. This will read the json file to determine what
  // type of client we're creating.
  RETURN_IF_NOT_OK(StorageClient::CreateStorageClient(this, schema_file_, &store_client_));

  // Perform the initial handshake with the storage client to further read the
  // dataset info to populate schema info and the number of rows in the client.
  RETURN_IF_NOT_OK(store_client_->LoadDatasetLayout());

  // Pull out the number of rows from the client and save into the op.
  num_rows_ = store_client_->num_rows();
  num_classes_ = store_client_->num_classes();

  return Status::OK();
}

// Init of the StorageOp.  This is 2 of 3 init.
// This version of the init allows the user to input the schema and other dataset properties rather
// than get it from the dataset itself.
Status StorageOp::InitOp(int32_t num_rows, const std::string &dataset_files_dir,
                         std::unique_ptr<DataSchema> data_schema, const std::string &labels_file_name,
                         const std::string &dataset_usage) {
  num_rows_ = num_rows;
  dataset_files_dir_ = dataset_files_dir;
  labels_file_name_ = labels_file_name;
  dataset_usage_ = dataset_usage;

  // Storage ops require the internal master/worker connector.  create it here
  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_conn_size_));

  // Get parameter for distribution.
  RETURN_IF_NOT_OK(LoadParallelConfig());

  // Create the storage client based on the dataset type given from the input schema.
  RETURN_IF_NOT_OK(StorageClient::CreateStorageClient(this, data_schema->dataset_type(), &store_client_));

  // Perform the initial handshake with the storage client to initialize the schema
  // and the number of rows in the set.  In this case, since the schema and the number
  // of rows is input by the user directly, it's not much of a "handshake", it's more
  // like an assign.
  RETURN_IF_NOT_OK(store_client_->AssignDatasetLayout(num_rows_, *data_schema));
  num_classes_ = store_client_->num_classes();

  return Status::OK();
}

// Init of the StorageOp.  This is 3 of 3 init.
// This version of the init does not take the schema in it's arguments. It must perform an
// internal handshake with the dataset to produce the schema.  Unlike constructor 1, it takes a
// list of files rather than a directory.
Status StorageOp::InitOp(const std::vector<std::string> &files_list, const std::string &schema_file) {
  dataset_file_list_ = files_list;
  schema_file_ = schema_file;

  // Storage ops require the internal master/worker connector.  create it here
  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_conn_size_));

  // Get parameter for distribution.
  RETURN_IF_NOT_OK(LoadParallelConfig());

  // Create the storage client. This will read the json file to determine what
  // type of client we're creating.
  RETURN_IF_NOT_OK(StorageClient::CreateStorageClient(this, schema_file_, &store_client_));

  // Perform the initial handshake with the storage client to further read the
  // dataset info to populate schema info and the number of rows in the client.
  RETURN_IF_NOT_OK(store_client_->LoadDatasetLayout());

  // Pull out the number of rows from the client and save into the op.
  num_rows_ = store_client_->num_rows();

  return Status::OK();
}

// Private helper method.  This one encapsulates some common construction/reset tasks and is
// designed to be re-entrant so that you can re-init a previously used StorageOp without needing
// to redo the storage client handshake.
Status StorageOp::init() {
  // First a sanity check to make sure the StorageClient initialization has done the proper
  // handshake and initialized both the schema and the number of rows for the dataset.
  if (store_client_->schema()->NumColumns() == 0 || num_rows_ == 0) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Storage client did not run handshake to init schema and number of rows.");
  }

  // If the data buffer vector is not empty, then we may be redoing a scan again after a repeat.
  // In such a case, we have vector of nullptrs that used to hold the buffers.  get rid of this
  // so we can reuse the vector.
  if (!data_buffers_.empty()) {
    data_buffers_.clear();
  }
  int32_t buffers_needed;

  // We have our range of row id's, but we must carve this up into buffers now so that
  // each buffer holds a subset of the overall range.
  // Instantiate the buffers now, but this does not actually drive a load of actual
  // data at this point.

  // First, compute how many buffers we would need to accomplish rowsPerBuffer
  buffers_needed = this->num_rows() / rows_per_buffer_;

  // If an extra partial buffer is needed, adjust for that.
  if (this->num_rows() % rows_per_buffer_ != 0) {
    buffers_needed++;
  }
  MS_LOG(INFO) << "Master: Initializing StorageOp. Dataset files dir: " << dataset_files_dir_ << " Dataset type: "
               << static_cast<std::underlying_type<DatasetType>::type>(store_client_->schema()->dataset_type())
               << " Dataset schema file: " << schema_file_ << " Number of rows: " << num_rows_
               << " Rows per buffer: " << rows_per_buffer_ << " Num buffers (computed): " << buffers_needed
               << " Number of workers: " << num_workers_ << ".";

  // Next, create each buffer in a loop.
  int32_t buff_id = 0;
  for (buff_id = 0; buff_id < buffers_needed; buff_id++) {
    // Create a new data buffer as a base class pointer, using the factory method from
    // DataBuffer class
    std::unique_ptr<DataBuffer> new_data_buffer;
    RETURN_IF_NOT_OK(DataBuffer::CreateDataBuffer(buff_id, store_client_, &new_data_buffer));

    // Insert the buffer into our vector
    data_buffers_.push_back(std::move(new_data_buffer));
  }

  // Instantiate the action queues.  If this was a re-entrant call then these already exist.
  // We cannot drop and recreate them because there are threads waiting on them currently.
  // They should be empty anyway in a reset codepath
  if (action_queue_.empty()) {
    // The max size of these queues should ensure they will never get full and they support
    // precisely the amount of data that we know they will hold (the total number of buffers).
    // There needs to be one queue for each worker, to support the Connector design for how
    // data will be fetched and pushed into a Connector in parallel.
    //
    // Say the total buffers is 5, and we have 2 workers.
    // To support this, we'd need 1 queue of size 2 and the other of size 3.
    // For simplicity, we'll make both of them 3 so they are the same size.
    int32_t action_queue_size = (buffers_needed / num_workers_) + 1;
    for (int32_t i = 0; i < num_workers_; ++i) {
      auto new_queue = std::make_unique<Queue<int32_t>>(action_queue_size);
      action_queue_.push_back(std::move(new_queue));
    }
  }

  // Extract the list of buffer id's from the vector and use this as our starting action
  // queue of buffers.
  RETURN_IF_NOT_OK(this->FillActionQueue(false));
  return Status::OK();
}

// Destructor
StorageOp::~StorageOp() {}

// A print method typically used for debugging
void StorageOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <StorageOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nDetailed operator printing has not been implemented for this op.\n\n";
  }
}

// Private helper method.  This one posts a control indicator for each worker thread to consume
// from the action queue.  When the worker pops this msg, it will shut itself down gracefully.
Status StorageOp::PostEndOfData() {
  MS_LOG(INFO) << "Master: Processed all of the buffers. Send end-of-data message to workers.";

  // For each worker we add the message so that they can all get the memo
  for (int32_t i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(action_queue_[i]->Add(kEndOfActions));
  }
  return Status::OK();
}

// Private helper method.  This one populates the action queue with the list of buffer ids.
Status StorageOp::FillActionQueue(bool randomize) {
  // We only support adding the new list of id's to the queue if we are sure the old list
  // of actions is already done.  This might change in the future though
  for (int32_t i = 0; i < num_workers_; ++i) {
    if (!(action_queue_[i]->empty())) {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                    "Attempt to get buffer id's into a queue, but the queue not empty!");
    }
  }
  if (!data_buffers_.empty()) {
    // Add buffer id's to the queue. Buffer id's in our vector are just numbers from 0 up, so
    // basically just a list of consecutive numbers starting from 0 (incremented by 1).
    // If randomize is requested, the list of id's will be jumbled up (so not consecutive
    // order)
    if (!randomize) {
      // Round robin of filling each worker with the buffer id's
      int32_t curr_worker = 0;
      for (int32_t i = 0; i < data_buffers_.size(); ++i) {
        RETURN_IF_NOT_OK(action_queue_[curr_worker]->Add(i));
        curr_worker++;
        if (curr_worker == num_workers_) {
          curr_worker = 0;
        }
      }
    } else {
      std::vector<int32_t> random_ids;
      int32_t i;
      for (i = 0; i < data_buffers_.size(); ++i) {
        random_ids.push_back(i);
      }
      uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(random_ids.begin(), random_ids.end(), std::default_random_engine(seed));

      // Round robin of filling each worker with the buffer id's from randomized list
      int32_t curr_worker = 0;
      for (i = 0; i < random_ids.size(); ++i) {
        RETURN_IF_NOT_OK(action_queue_[curr_worker]->Add(random_ids[i]));
        curr_worker++;
        if (curr_worker == num_workers_) {
          curr_worker = 0;
        }
      }
    }
  }
  return Status::OK();
}

// The entry point code for when workers are launched.
// Given the input bufferId, it returns a shared_ptr to that buffer back to you by driving a
// load operation.  This function is intended to be run by worker threads, when they are
// populating the memory with the actual data of the buffer.
Status StorageOp::GetBuffer(int32_t buffer_id, std::unique_ptr<DataBuffer> *ptr) {
  if (!data_buffers_.empty()) {
    if (static_cast<size_t>(buffer_id) >= data_buffers_.size()) {
      std::ostringstream ss;
      ss << "Error.  Buffer id " << buffer_id << " is out of range.";
      std::string err_msg = ss.str();
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // execute a load operation to fill this buffer (may result in call to storage layers)
    RETURN_IF_NOT_OK(data_buffers_[buffer_id]->Load());

    // Return the buffer
    // Important: The share pointer remains counted for the caller as well as locally in the
    // mDataBuffers array.  Later when the buffer is sent on it's way up the pipeline, the
    // shared_ptr in the array will be reset so that the StorageOp will not hang on to old
    // buffers that it has already passed up the pipeline.
    *ptr = std::move(data_buffers_[buffer_id]);
  } else {
    RETURN_STATUS_UNEXPECTED("Requested to get a buffer from an empty cache.");
  }
  return Status::OK();
}

// Class functor operator () override.
// All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work
Status StorageOp::operator()() {
  // Before we enter our master loop, kick off our workers and assign them to
  // use the StorageOp worker entry code.
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&StorageOp::WorkerEntry, this, std::placeholders::_1)));
  // Handshake with TaskManager to synchronize thread creation
  TaskManager::FindMe()->Post();
  int32_t num_buffers_to_fetch = data_buffers_.size();

  // The storage op is the bottom node in the tree, so it does not listen to an input
  // queue from an operator below us. Instead, we'll will read from the internal queue
  // that our workers produce into, and then push that into output queue.
  bool done = false;
  std::unique_ptr<DataBuffer> fetched_buffer;
  while (!done) {
    // Get the next buffer. We are single thread master so thread id hard coded to 0
    // on the connector pop.  Count this buffer towards our count, and then push
    // it up to the output connector.
    RETURN_IF_NOT_OK(worker_connector_->PopWithRetry(0, &fetched_buffer));
    buffers_fetched_++;
    int32_t buffer_id = fetched_buffer->id();

    if (buffers_fetched_ == 1) {
      num_buffers_to_fetch = static_cast<int32_t>(data_buffers_.size());
    }

    // There should be 2 holders of this buffer currently. We have one in the mDataBuffers
    // table, and then ourselves right now with fetchedBuffer.
    // Reduce the shared_ptr ref count of this buffer by removing it from the mDataBuffers
    // table first before we push the buffer to output connector.
    data_buffers_[buffer_id].reset();
    MS_LOG(INFO) << "StorageOp master: Consumed buffer " << buffer_id << " from internal worker connector.";
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(fetched_buffer)));
    MS_LOG(INFO) << "StorageOp master: pushed buffer " << buffer_id << " to output connector.";

    // Now, check our loop exit conditions and perform appropriate end of data handling if
    // we've reached the end of our scan.
    if (buffers_fetched_ == num_buffers_to_fetch) {
      MS_LOG(INFO) << "StorageOp master: Reached end of data.";

      // If we are not inside of a Repeat path in the tree, or we are in a repeat path but
      // this was our last repeat, then we do a full quit here with eof control message.
      if (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat)) {
        // Post the control message to tell the workers to stop waiting on action queue
        // because we are done!
        RETURN_IF_NOT_OK(this->PostEndOfData());
        std::unique_ptr<DataBuffer> eoeBuffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoeBuffer)));
        MS_LOG(INFO) << "StorageOp master: Flow end-of-data eof message.";
        std::unique_ptr<DataBuffer> eofBuffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eofBuffer)));
        MS_LOG(INFO) << "StorageOp master: Main execution loop complete.";
        done = true;  // while loop exit
      } else {
        // We are in a repeat path and it's not the last repeat.
        // Flow an end-of-epoch control message up the pipeline.
        // RepeatOp above us somewhere in the tree will re-init us with the data to fetch again
        // once it gets the end-of-epoch message.
        MS_LOG(INFO) << "StorageOp master: Flow end-of-epoch eoe message.";
        std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));

        // reset our buffer count and go to loop again.
        buffers_fetched_ = 0;

        // This is a bit of a cheat.  Only the repeat op should perform resetting actions
        // against us (currently).  However, if we go to block/wait on the worker_connector_
        // right now before the reset is done (driven from the repeat op), then we end
        // up using stale connector index info and blocking on the wrong thing, causing
        // invalid order during the next epoch.
        // For now then, do a quick reset of just the connector queue so that we block
        // at a safe starting point in the connector.
        worker_connector_->Reset();
      }
    }
  }
  return Status::OK();
}

// The entry point code for when workers are launched.
Status StorageOp::WorkerEntry(int32_t worker_id) {
  int32_t next_action_id = 0;
  MS_LOG(INFO) << "Worker: StorageOp worker entry point.";

  // Handshake with TaskManager to synchronize the creation
  TaskManager::FindMe()->Post();

  // While there is still some actions to perform
  RETURN_IF_NOT_OK(action_queue_[worker_id]->PopFront(&next_action_id));
  while (next_action_id != kEndOfActions) {
    // Drive a load of this buffer and get a pointer to the buffer after it's loaded in
    std::unique_ptr<DataBuffer> dB;
    RETURN_IF_NOT_OK(this->GetBuffer(next_action_id, &dB));
    MS_LOG(INFO) << "Worker: Loaded buffer " << next_action_id << ".";

    // Add the buffer to the internal queue for master to consume from later.
    // This could end up blocking if the queue is full in which case it waits here
    // until the master can drain a buffer off the queue.
    RETURN_IF_NOT_OK(worker_connector_->Add(worker_id, std::move(dB)));
    MS_LOG(INFO) << "Worker: Pushed buffer " << next_action_id << " to internal worker connector.";

    // Get the next action id and loop
    RETURN_IF_NOT_OK(action_queue_[worker_id]->PopFront(&next_action_id));
  }
  MS_LOG(INFO) << "Worker: Received end-of-data message.  Worker complete.";
  return Status::OK();
}

const DataSchema *StorageOp::schema() const { return store_client_->schema(); }

// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status StorageOp::Reset() {
  RETURN_IF_NOT_OK(ParallelOp::Reset());  // Call our super class reset first.

  // We do not need to redo the handshake with the storage client, since that
  // info should be the same as the last time.  However there may be stale
  // state info in the client from the last execution.  The client provides
  // a reset method as well to re-initialize.
  RETURN_IF_NOT_OK(store_client_->Reset());

  // init method is re-entrant and will refresh everything.
  RETURN_IF_NOT_OK(this->init());
  return Status::OK();
}

// Name: LoadParallelConfig
// Description: Load parallel config info from a specific config file. In multi-P cases (or single-P cases), we
//             need to know deviceID, rank, device number, shard mode
//             , shuffle (or not) and seed to prepare to scatter files.
Status StorageOp::LoadParallelConfig() {
  if (data_distribution_file_ == "") {
    return Status::OK();
  }
  try {
    std::ifstream in(data_distribution_file_);
    nlohmann::json js;
    in >> js;
    device_num_ = js.value("deviceNum", 0);
    device_id_ = js.value("deviceId", 0);
    if (device_num_ == 0 || device_num_ > MAX_INTEGER_INT32) {
      RETURN_STATUS_UNEXPECTED("Invalid deviceNum");
    }
    if (device_id_ > MAX_INTEGER_INT32 || device_id_ >= device_num_) {
      MS_LOG(INFO) << "In parallel config file " << data_distribution_file_ << ", wrong deviceID provided.";
      RETURN_STATUS_UNEXPECTED("Invalid deviceId");
    }
    shard_config_ = js.value("shardConfig", "");
    if (shard_config_ != "ALL" && shard_config_ != "UNIQUE" && shard_config_ != "RANDOM") {
      MS_LOG(INFO) << "In parallel config file " << data_distribution_file_ << " wrong mShardConfig provided.";
      RETURN_STATUS_UNEXPECTED("Invalid shardConfig");
    }
    std::string shuffle_str = js.value("shuffle", "");
    if (shuffle_str == "ON") {
      shuffle_config_ = true;
    } else if (shuffle_str == "OFF") {
      shuffle_config_ = false;
    } else {
      MS_LOG(INFO) << "In parallel config file " << data_distribution_file_
                   << ", shuffle config is wrong: it's not ON or OFF";
      RETURN_STATUS_UNEXPECTED("Invalid shuffle option");
    }
    seed_ = js.value("seed", 0);
    if (seed_ > MAX_INTEGER_UINT32) {
      RETURN_STATUS_UNEXPECTED("Invalid seed");
    }
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Load parallel config failed");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
