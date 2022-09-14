/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <map>
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
  ParallelOp(int32_t num_workers, int32_t op_connector_size, const std::shared_ptr<SamplerRT> sampler = nullptr)
      : DatasetOp(op_connector_size, sampler),
        num_workers_paused_(0),
        epoch_sync_flag_(false),
        num_workers_(num_workers),
        next_worker_id_(0),
        worker_connector_size_(op_connector_size) {
    // reduce excessive memory usage with high parallelism
    constexpr int32_t worker_limit = 4;
    if (num_workers_ > worker_limit) {
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

  Status WaitForWorkers() override {
    // reset num_paused workers to 0
    num_workers_paused_ = 0;
    for (int32_t wkr_id = 0; wkr_id < num_workers_; wkr_id++) {
      RETURN_IF_NOT_OK(SendWaitFlagToWorker(NextWorkerID()));
    }
    // wait until all workers are done processing their work in local_queue_
    RETURN_IF_NOT_OK(wait_for_workers_post_.Wait());
    next_worker_id_ = 0;
    // clear the WaitPost for the next Wait()
    wait_for_workers_post_.Clear();
    return Status::OK();
  }

  /// Add a new worker to the parallelOp. The function will have to wait for all workers to process current rows.
  /// Then it adds a new thread to the list.
  /// \note The caller of this function has to be the main thread of the Op, since it's the only entity responsible to
  /// push rows to workers_in_queue
  /// \return Status The status code returned
  Status AddNewWorkers(int32_t num_new_workers = 1) override {
    // wait for workers to process the current rows
    RETURN_IF_NOT_OK(WaitForWorkers());
    for (int32_t i = 0; i < num_new_workers; i++) {
      RETURN_IF_NOT_OK(worker_in_queues_.AddQueue(tree_->AllTasks()));
      RETURN_IF_NOT_OK(worker_out_queues_.AddQueue(tree_->AllTasks()));
      Task *new_task;
      RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
        Name() + "::WorkerEntry", std::bind(&ParallelOp::WorkerEntry, this, num_workers_), &new_task, id()));
      CHECK_FAIL_RETURN_UNEXPECTED(new_task != nullptr, "Cannot create a new worker.");
      worker_tasks_.push_back(new_task);
      num_workers_++;
      MS_LOG(INFO) << "A new worker has been added to op: " << Name() << "::" << id()
                   << " num_workers=" << num_workers_;
    }
    return Status::OK();
  }

  /// Add a new worker to the parallelOp. The function will have to wait for all workers to process current rows.
  /// Then it adds a new thread to the list.
  /// \note The caller of this function has to be the main thread of the Op, since it's the only entity responsible to
  /// push rows to workers_in_queue
  /// \return Status The status code returned
  Status RemoveWorkers(int32_t num_workers = 1) override {
    // wait for workers to process the current rows
    RETURN_IF_NOT_OK(WaitForWorkers());
    for (size_t i = 0; i < num_workers; i++) {
      RETURN_IF_NOT_OK(SendQuitFlagToWorker(static_cast<size_t>(num_workers_) - 1));
      RETURN_IF_NOT_OK(worker_tasks_[static_cast<size_t>(num_workers_) - 1]->Join());
      RETURN_IF_NOT_OK(worker_in_queues_.RemoveLastQueue());
      worker_tasks_.pop_back();
      num_workers_--;
      MS_LOG(INFO) << "Worker ID " << num_workers_ << " is requested to be removed in operator: " << NameWithID()
                   << " num_workers=" << num_workers_;
    }
    return Status::OK();
  }

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

    RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_,
                                          std::bind(&ParallelOp::WorkerEntry, this, std::placeholders::_1),
                                          &worker_tasks_, Name() + "::WorkerEntry", id()));
    RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&ParallelOp::Collector, this), Name() + "::Collector", id()));

    return Status::OK();
  }

  virtual Status Collector() {
    TaskManager::FindMe()->Post();
    // num_rows received, including eoe, num_step of current epoch
    int64_t num_rows = 0, ep_step = 0, total_step = 0;
    int32_t current_repeats = 0, current_epochs = 0;
    TensorRow row;
    do {
      RETURN_IF_NOT_OK(worker_out_queues_[static_cast<const int>(num_rows++ % num_workers_)]->PopFront(&row));
      if (row.wait()) {
        // When collector receives the signal from workere thread, it increments a atomic int
        // If num_worker signals are received, wakes up the main thread
        if (++num_workers_paused_ == num_workers_) {
          wait_for_workers_post_.Set();
          num_rows = 0;
        }
        continue;
      } else if (row.eoe()) {
        current_repeats++;
        // check whether this is the end of a real epoch (not all eoe signals end of epoch)
        if (current_repeats % GetOpNumRepeatsPerEpoch() == 0) {
          current_epochs++;
          RETURN_IF_NOT_OK(callback_manager_.EpochEnd(CallbackParam(current_epochs, ep_step, total_step)));
          ep_step = 0;
        }
      } else if (row.eof()) {
        RETURN_IF_NOT_OK(callback_manager_.End(CallbackParam(current_epochs + 1, ep_step, total_step)));
      } else if (row.skip()) {
        continue;
      } else if (row.Flags() == TensorRow::TensorRowFlags::kFlagNone) {
        ++ep_step;
        ++total_step;
        RETURN_IF_NOT_OK(callback_manager_.StepEnd(CallbackParam(current_epochs + 1, ep_step, total_step)));
      }
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(row)));
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

  std::vector<Task *> worker_tasks_;

  int32_t NextWorkerID() {
    int32_t next_worker = next_worker_id_;
    next_worker_id_ = (next_worker_id_ + 1) % num_workers_;
    return next_worker;
  }

 public:
  int32_t NumWorkers() override { return num_workers_; }

 protected:
  std::atomic_int next_worker_id_;

  std::map<int32_t, std::atomic_bool> quit_ack_;

  /// The size of input/output worker queeus
  int32_t worker_connector_size_;
  /// queues to hold the input rows to workers
  QueueList<T> worker_in_queues_;
  /// queues to hold the output from workers
  QueueList<S> worker_out_queues_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_
