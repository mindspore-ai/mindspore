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
#include "dataset/engine/datasetops/parallel_op.h"

#include <iostream>
#include <utility>
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/core/config_manager.h"
#include "dataset/engine/db_connector.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Constructor
ParallelOp::ParallelOp(int32_t num_workers, int32_t op_connector_size)
    : DatasetOp(op_connector_size),
      num_workers_(num_workers),
      num_producers_(num_workers),
      worker_connector_size_(1),
      worker_connector_(nullptr) {}

// Creates the internal worker connector for the parallel op if the derived class wants to use it
Status ParallelOp::CreateWorkerConnector(int32_t worker_connector_size) {
  if (worker_connector_size == 0) {
    RETURN_STATUS_UNEXPECTED("Worker connector size 0 is invalid.");
  }
  num_producers_ = 1;
  worker_connector_size_ = worker_connector_size;
  // Instantiate the worker connector.  This is the internal connector, not the operators
  // output connector.  It has single master consuming from it (num producers is 1), and the number
  // of workers is the defined count from the op.
  worker_connector_ = std::make_unique<DbConnector>(num_workers_, num_producers_, worker_connector_size);

  return Status::OK();
}

// A print method typically used for debugging
void ParallelOp::Print(std::ostream &out, bool show_all) const {
  // Summary 1-liner print
  if (!show_all) {
    out << " [workers: " << num_workers_ << "]";
    // Call super class printer
    DatasetOp::Print(out, show_all);
  } else {
    // Detailed print
    DatasetOp::Print(out, show_all);
    out << "\nNum workers: " << num_workers_;
  }
}

// Override base class reset to provide reset actions specific to the ParallelOp class.
Status ParallelOp::Reset() {
  RETURN_IF_NOT_OK(DatasetOp::Reset());  // Perform any super class reset work

  // ParallelOp is abstract, but we do own the connector between workers and master
  // (if the parallel op is configured for this).  Reset that connector here.
  if (worker_connector_) {
    worker_connector_->Reset();
  }

  return Status::OK();
}

// Register the internal worker connectors
Status ParallelOp::RegisterWorkerConnectors() {
  if (worker_connector_) {
    return (worker_connector_->Register(tree_->AllTasks()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
