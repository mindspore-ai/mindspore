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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// global const in our namespace
constexpr int32_t kEndOfActions = -1;

// Forward declares
class DataBuffer;

class DbConnector;

// A ParallelOp provides a multi-threaded DatasetOp
class ParallelOp : public DatasetOp {
 public:
  // Constructor
  // @param num_workers
  // @param op_connector_size - size of the output connector for this operator
  // @param sampler - The sampler for the op
  ParallelOp(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler = nullptr);

  // Destructor
  ~ParallelOp() = default;

  // Creates the internal worker connector for the parallel op if the derived class wants to use it.
  // @notes This changes the number of producers of this op to 1, since it establishes a master/worker
  // relationship within the op, making all production flow through a single master.
  // @return Status - The error return code
  Status CreateWorkerConnector(int32_t worker_connector_size);

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kParallelOp; }

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param pO - reference to the ParallelOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ParallelOp &po) {
    po.Print(out, false);
    return out;
  }

  // Override base class reset to provide reset actions specific to the ParallelOp class.
  // @return Status The status code returned
  Status Reset() override;

  // Getter
  // @return the number of workers
  int32_t num_workers() const override { return num_workers_; }

  // Getter
  // @return the number of threads consuming from the previous Connector
  int32_t num_consumers() const override { return num_workers_; }

  // Getter
  // @return the number of producers pushing to the output Connector
  // @notes The number of producers is commonly the same as number of workers, except in the case
  // when a worker connector is set up.  In that case, there are n workers, and a single master
  // such that only 1 thread is a producer rather than the n workers.
  // @return the number of producers
  int32_t num_producers() const override { return num_producers_; }

  // Register the internal worker connectors.
  // @return Status
  Status RegisterWorkerConnectors() override;

 protected:
  // Interface for derived classes to implement. All derived classes must provide the entry
  // function with the main execution loop for worker threads.
  // @return Status The status code returned
  virtual Status WorkerEntry(int32_t workerId) = 0;

  /// This function is only intended to be called by CallbackManager within the master thread of ParallelOp
  /// The expected behavior is this, when this function is invoked, this function will block until all the workers
  /// have finished their remaining work and go to sleep. Since all ParallelOps use a QueueList to sync with master.
  /// They would automatically wait on the QueueList when they are done.
  /// \return Status
  Status WaitForWorkers() override;

  // Wait post used to perform the pausing logic
  WaitPost wait_for_workers_post_;

  // Count number of workers that have signaled master
  std::atomic_int num_workers_paused_;

  // Whether or not to sync worker threads at the end of each epoch
  bool epoch_sync_flag_;

  int32_t num_workers_;    // The number of worker threads
  int32_t num_producers_;  // The number of threads pushing to the out_connector_
  int32_t worker_connector_size_;
  std::unique_ptr<DbConnector> worker_connector_;        // The internal connector for worker threads
  QueueList<std::unique_ptr<IOBlock>> io_block_queues_;  // queues of IOBlocks
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_
