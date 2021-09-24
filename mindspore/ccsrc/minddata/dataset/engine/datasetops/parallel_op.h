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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class ExecutionTree;

// A ParallelOp provides a multi-threaded DatasetOp
template <typename T, typename S>
class ParallelOp : public DatasetOp {
 public:
  /// Constructor
  /// \param num_workers
  /// \param op_connector_size - size of the output connector for this operator
  /// \param sampler - The sampler for the op
  ParallelOp(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler = nullptr)
      : DatasetOp(op_connector_size, sampler),
        num_workers_(num_workers),
        worker_connector_size_(op_connector_size),
        num_workers_paused_(0),
        epoch_sync_flag_(false) {
    // reduce excessive memory usage with high parallelism
    // when num_workers > 4, reduce op_connector_size to have similar total size if there were only 4 workers
    constexpr int32_t worker_limit = 4;
    if (num_workers_ > worker_limit) {
      oc_queue_size_ = std::max(1, op_connector_size * worker_limit / num_workers_);
      worker_connector_size_ = std::max(1, op_connector_size * worker_limit / num_workers_);
    }
  }
  // Destructor
  ~ParallelOp() = default;

  /// A print method typically used for debugging
  /// \param out - The output stream to write output to
  /// \param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override {
    DatasetOp::Print(out, show_all);
    out << " [workers: " << num_workers_ << "]";
  }

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

  int32_t NumWorkers() const override { return num_workers_; }

  // Getter
  // @return the number of threads consuming from the previous Connector
  int32_t NumConsumers() const override { return num_workers_; }

  // Getter
  // @return the number of producers pushing to the output Connector
  // @notes The number of producers is commonly the same as number of workers, except in the case
  // when a worker connector is set up.  In that case, there are n workers, and a single master
  // such that only 1 thread is a producer rather than the n workers.
  // @return the number of producers
  int32_t NumProducers() const override { return num_producers_; }

 protected:
  /// Interface for derived classes to implement. All derived classes must provide the entry
  /// function with the main execution loop for worker threads.
  /// \return Status The status code returned
  virtual Status WorkerEntry(int32_t workerId) = 0;

  /// Called first when function is called
  /// \return Status The status code returned
  virtual Status RegisterAndLaunchThreads() {
    RETURN_UNEXPECTED_IF_NULL(tree_);
    worker_in_queues_.Init(num_workers_, worker_connector_size_);
    worker_out_queues_.Init(num_workers_, worker_connector_size_);

    // Registers QueueList and individual Queues for interrupt services
    RETURN_IF_NOT_OK(worker_in_queues_.Register(tree_->AllTasks()));
    RETURN_IF_NOT_OK(worker_out_queues_.Register(tree_->AllTasks()));
    RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));

    RETURN_IF_NOT_OK(tree_->LaunchWorkers(
      num_workers_, std::bind(&ParallelOp::WorkerEntry, this, std::placeholders::_1), Name() + "::WorkerEntry", id()));
    RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&ParallelOp::Collector, this), Name() + "::Collector", id()));

    return Status::OK();
  }

  virtual Status Collector() {
    TaskManager::FindMe()->Post();
    uint64_t ctr = 0;
    TensorRow row;
    do {
      RETURN_IF_NOT_OK(worker_out_queues_[ctr++ % num_workers_]->PopFront(&row));
      if (row.eoe() || row.eof() || !row.empty()) {
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(row)));
      }
    } while (!row.eof());
    return Status::OK();
  }

  // Wait post used to perform the pausing logic
  WaitPost wait_for_workers_post_;

  // Count number of workers that have signaled master
  std::atomic_int num_workers_paused_;

  /// Whether or not to sync worker threads at the end of each epoch
  bool epoch_sync_flag_;

  /// The number of worker threads
  int32_t num_workers_;
  int32_t num_producers_;  // The number of threads pushing to the out_connector_
  /// The size of input/output worker queeus
  int32_t worker_connector_size_;
  /// queues to hold the input rows to workers
  QueueList<T> worker_in_queues_;
  /// queues to hold the output of workers
  QueueList<S> worker_out_queues_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_
