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
#include <deque>
#include <map>
#include <memory>
#include <mutex>
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
constexpr int64_t kCachedRowsSize = 16;

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
        worker_connector_size_(op_connector_size),
        strategy_{nullptr} {
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

  int32_t NumWorkers() const override {
    int32_t num_workers = 1;
    {
      std::unique_lock<std::mutex> _lock(mux_);
      num_workers = num_workers_;
    }
    return num_workers;
  }

  // pause all the worker thread and collector thread
  Status WaitForWorkers() override {
    // reset num_paused workers to 0
    num_workers_paused_ = 0;
    uint32_t num_workers = NumWorkers();
    for (int32_t wkr_id = 0; wkr_id < num_workers; wkr_id++) {
      RETURN_IF_NOT_OK(SendWaitFlagToWorker(NextWorkerID()));
    }
    // wait until all workers are done processing their work in local_queue_
    RETURN_IF_NOT_OK(wait_for_workers_post_.Wait());
    next_worker_id_ = 0;
    // clear the WaitPost for the next Wait()
    wait_for_workers_post_.Clear();
    return Status::OK();
  }

  // wakeup all the worker threads and collector thread
  Status PostForWorkers() override {
    // wakeup old workers
    for (auto &item : worker_tasks_) {
      item->Post();
    }

    // wakeup the collector thread
    wait_for_collector_.Set();

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
    }

    for (int32_t i = 0; i < num_new_workers; i++) {
      Task *new_task;
      RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
        Name() + "::WorkerEntry", std::bind(&ParallelOp::WorkerEntry, this, num_workers_), &new_task, id()));
      CHECK_FAIL_RETURN_UNEXPECTED(new_task != nullptr, "Cannot create a new worker.");
      worker_tasks_.push_back(new_task);
      {
        std::unique_lock<std::mutex> _lock(mux_);
        num_workers_++;
      }
      MS_LOG(INFO) << "A new worker has been added to op: " << Name() << "::" << id()
                   << " num_workers=" << num_workers_;
    }

    // wakeup all the workers threads and collector thread
    RETURN_IF_NOT_OK(PostForWorkers());

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
      worker_tasks_[num_workers_ - 1]->Post();  // wakeup the worker
      RETURN_IF_NOT_OK(worker_tasks_[static_cast<size_t>(num_workers_) - 1]->Join());
      RETURN_IF_NOT_OK(worker_in_queues_.RemoveLastQueue());
      worker_tasks_.pop_back();
      {
        std::unique_lock<std::mutex> _lock(mux_);
        num_workers_--;
      }
      MS_LOG(INFO) << "Worker ID " << num_workers_ << " is requested to be removed in operator: " << NameWithID()
                   << " num_workers=" << num_workers_;
    }

    // wakeup all the workers threads and collector thread
    RETURN_IF_NOT_OK(PostForWorkers());

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

  class RowHandlingStrategy {
   public:
    explicit RowHandlingStrategy(ParallelOp *op) : op_(op) {}
    virtual ~RowHandlingStrategy() = default;

    virtual Status HandleHealthyRow([[maybe_unused]] TensorRow *row) {
      ++this->op_->ep_step_;
      ++this->op_->total_step_;
      RETURN_IF_NOT_OK(this->op_->callback_manager_.StepEnd(CallbackParam(
        static_cast<int64_t>(this->op_->current_epochs_) + 1, this->op_->ep_step_, this->op_->total_step_)));
      return this->op_->out_connector_->Add(std::move(*row));
    }
    virtual Status HandleErrorRow([[maybe_unused]] TensorRow *row) = 0;

    virtual Status HandleEOE([[maybe_unused]] TensorRow *row) {
      this->op_->current_repeats_++;
      // check whether this is the end of a real epoch (not all eoe signals end of epoch)
      if (this->op_->current_repeats_ % this->op_->GetOpNumRepeatsPerEpoch() == 0) {
        this->op_->current_epochs_++;
        RETURN_IF_NOT_OK(this->op_->callback_manager_.EpochEnd(
          CallbackParam(this->op_->current_epochs_, this->op_->ep_step_, this->op_->total_step_)));
        this->op_->ep_step_ = 0;
      }
      return op_->out_connector_->Add(std::move(*row));
    }
    virtual Status HandleEOF([[maybe_unused]] TensorRow *row) {
      RETURN_IF_NOT_OK(this->op_->callback_manager_.End(CallbackParam(
        static_cast<int64_t>(this->op_->current_epochs_) + 1, this->op_->ep_step_, this->op_->total_step_)));
      return op_->out_connector_->Add(std::move(*row));
    }

   protected:
    ParallelOp *op_;
  };

  class ErrorStrategy : public RowHandlingStrategy {
   public:
    using RowHandlingStrategy::RowHandlingStrategy;
    Status HandleErrorRow([[maybe_unused]] TensorRow *row) override {
      return Status(StatusCode::kMDUnexpectedError,
                    "[Internal Error] Error row is detected in collector while Error strategy is set to error out!");
    }
  };

  class SkipStrategy : public RowHandlingStrategy {
   public:
    using RowHandlingStrategy::RowHandlingStrategy;
    Status HandleErrorRow([[maybe_unused]] TensorRow *row) override { return Status::OK(); }
  };

  class ReplaceStrategy : public RowHandlingStrategy {
   public:
    using RowHandlingStrategy::RowHandlingStrategy;

    Status HandleHealthyRow([[maybe_unused]] TensorRow *row) override {
      CHECK_FAIL_RETURN_UNEXPECTED(backup_index_ < kCachedRowsSize,
                                   "[Internal Error] Number of cached rows is beyond the number set.");
      if (backup_index_ < kCachedRowsSize - 1) {  // cache has used row(s) or is not full
        if (IsCacheFull()) {
          // remove the last element from cache (a used row)
          PopFromCache();
        }
        RETURN_IF_NOT_OK(AddToCache(*row));
      } else {  // cache is full of unused rows
        if (missing_errors_ > 0) {
          // send a cached row to next op and cache the current row
          RETURN_IF_NOT_OK(AddFromCache());
          PopFromCache();
          missing_errors_--;
          RETURN_IF_NOT_OK(AddToCache(*row));
        }
      }
      // send the healthy row to next op
      ++this->op_->ep_step_;
      ++this->op_->total_step_;
      RETURN_IF_NOT_OK(this->op_->callback_manager_.StepEnd(CallbackParam(
        static_cast<int64_t>(this->op_->current_epochs_) + 1, this->op_->ep_step_, this->op_->total_step_)));
      return this->op_->out_connector_->Add(std::move(*row));
    }

    Status HandleErrorRow([[maybe_unused]] TensorRow *row) override {
      CHECK_FAIL_RETURN_UNEXPECTED(backup_index_ < kCachedRowsSize,
                                   "[Internal Error] Number of cached rows is beyond the number set.");
      // cache is not full of unused rows
      if (backup_index_ != kCachedRowsSize - 1) {
        missing_errors_++;
        return Status::OK();
      }
      // cache is full of unused rows and we have an error row
      return AddFromCache();
    }

    Status HandleEOE([[maybe_unused]] TensorRow *row) override {
      CHECK_FAIL_RETURN_UNEXPECTED(missing_errors_ == 0 || !IsCacheEmpty(),
                                   "All data is garbage and cannot be replaced.");
      // send outstanding rows first and then send eoe
      while (missing_errors_ > 0) {
        RETURN_IF_NOT_OK(AddFromCache());
        missing_errors_--;
      }
      return RowHandlingStrategy::HandleEOE(row);
    }

    Status HandleEOF([[maybe_unused]] TensorRow *row) override {
      // release memory
      std::deque<TensorRow>().swap(backup_rows);
      return RowHandlingStrategy::HandleEOF(row);
    }

   private:
    Status AddFromCache() {
      CHECK_FAIL_RETURN_UNEXPECTED(backup_rows.size() > 0, "Cannot add a row from cache since cache is empty!");
      const TensorRow &cached_row = backup_rows[static_cast<size_t>(backup_index_) % backup_rows.size()];
      TensorRow copy_row;
      RETURN_IF_NOT_OK(cached_row.Clone(&copy_row));
      backup_index_--;
      ++this->op_->ep_step_;
      ++this->op_->total_step_;
      RETURN_IF_NOT_OK(this->op_->callback_manager_.StepEnd(CallbackParam(
        static_cast<int64_t>(this->op_->current_epochs_) + 1, this->op_->ep_step_, this->op_->total_step_)));
      return this->op_->out_connector_->Add(std::move(copy_row));
    }

    Status AddToCache(const TensorRow &row) {
      CHECK_FAIL_RETURN_UNEXPECTED(backup_rows.size() < kCachedRowsSize,
                                   "[Internal Error] Inserting another row to cache while cache is already full.");
      CHECK_FAIL_RETURN_UNEXPECTED(
        backup_index_ < kCachedRowsSize - 1,
        "[Internal Error] Inserting another row to cache while cache is already full of unused rows.");
      TensorRow copy_row;
      RETURN_IF_NOT_OK(row.Clone(&copy_row));
      (void)backup_rows.emplace_front(std::move(copy_row));
      backup_index_++;
      return Status::OK();
    }

    void PopFromCache() { backup_rows.pop_back(); }
    bool IsCacheFull() const { return backup_rows.size() == kCachedRowsSize; }
    bool IsCacheEmpty() const { return backup_rows.size() == 0; }
    std::deque<TensorRow> backup_rows{};  // will hold a copy of some healthy rows collected (NOT error, skip, eoe, eof)
    int32_t backup_index_{-1};  // index of the backup we should pick next time (can be negative if we run out of
    // unused cached rows)
    int32_t missing_errors_{0};  // the number of unaddressed error rows (that we need to send a replacement to output)
  };

  virtual Status Collector() {
    TaskManager::FindMe()->Post();
    // num_rows received, including eoe,
    int64_t num_rows = 0;
    current_repeats_ = 0;
    current_epochs_ = 0;
    SetStrategy();
    // num_step of current epoch and the total
    ep_step_ = 0, total_step_ = 0;
    do {
      TensorRow row;
      RETURN_IF_NOT_OK(worker_out_queues_[static_cast<const int>(num_rows++ % NumWorkers())]->PopFront(&row));
      if (row.wait()) {
        // When collector receives the signal from worker thread, it increments an atomic int
        // If num_worker signals are received, wakes up the main thread
        if (++num_workers_paused_ == num_workers_) {
          wait_for_workers_post_.Set();
          RETURN_IF_NOT_OK(wait_for_collector_.Wait());
          wait_for_collector_.Clear();
          num_rows = 0;
        }
        continue;
      } else if (row.eoe()) {
        RETURN_IF_NOT_OK(strategy_->HandleEOE(&row));
      } else if (row.eof()) {
        RETURN_IF_NOT_OK(strategy_->HandleEOF(&row));
        break;
      } else if (row.skip()) {
        continue;
      } else if (row.error()) {
        RETURN_IF_NOT_OK(strategy_->HandleErrorRow(&row));
      } else if (row.Flags() == TensorRow::TensorRowFlags::kFlagNone) {
        RETURN_IF_NOT_OK(strategy_->HandleHealthyRow(&row));
      }
    } while (true);
    return Status::OK();
  }

  // Wait post used to perform the pausing logic
  WaitPost wait_for_workers_post_;

  // Wait post used to perform the collector thread
  WaitPost wait_for_collector_;

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
  int32_t NumWorkers() override {
    int32_t num_workers = 1;
    {
      std::unique_lock<std::mutex> _lock(mux_);
      num_workers = num_workers_;
    }
    return num_workers;
  }

 private:
  void SetStrategy() {
    if (Name() != kMapOp) {
      strategy_ = std::make_unique<ErrorStrategy>(this);
      return;
    }
    if (GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kSkip) {
      strategy_ = std::make_unique<SkipStrategy>(this);
    } else if (GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kReplace) {
      strategy_ = std::make_unique<ReplaceStrategy>(this);
    } else {
      strategy_ = std::make_unique<ErrorStrategy>(this);
    }
  }

 protected:
  std::atomic_int next_worker_id_;

  std::map<int32_t, std::atomic_bool> quit_ack_;

  /// The size of input/output worker queeus
  int32_t worker_connector_size_;
  /// queues to hold the input rows to workers
  QueueList<T> worker_in_queues_;
  /// queues to hold the output from workers
  QueueList<S> worker_out_queues_;

  // lock for num_workers_ read and write
  mutable std::mutex mux_;

 private:
  std::unique_ptr<RowHandlingStrategy> strategy_;
  int32_t ep_step_{0};
  int32_t total_step_{0};
  int32_t current_epochs_{0};
  int32_t current_repeats_{0};
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PARALLEL_OP_H_
