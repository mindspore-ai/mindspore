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
#ifndef DATASET_ENGINE_CONNECTOR_H_
#define DATASET_ENGINE_CONNECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dataset/util/task_manager.h"
#include "dataset/util/queue.h"
#include "dataset/util/services.h"
#include "dataset/util/cond_var.h"

namespace mindspore {
namespace dataset {
// Connector is a communication data structure between two group of threads that
// preserve the order.
//
// Example use case:
// An initial tasks-list of [1,2,3,4,5,6,7,8,9] with 5 threads getting/processing elements from that list,
// and pushing the processed elements to a Connector in any order whoever finishes processing first.
// If the consumer of the Connector is single threaded, when the consumer pop() the
// element from the Connector one by one, it will get [1,2,3,4,5,6,7,8,9].
//
// Requirements:
//   1. Each thread in the group of consumer or producer threads must be assigned ids starting from 0.
//   2. If your multi-threads program is not reading from a Connector class but
//      want to push to a Connector class, you must follow roundrobin element distribution,
//      i.e., the thread-id0 must have the first element, thread-id1 has the second element,
//      and so on; then each of this worker can push to the Connector class async in parallel.
//
// Blocking conditions:
//   1. Connector.push(int, T) can block when the internal queue it's trying to push is full.
//   2. Connector.pop(int) can block when
//        - The internal queue it's trying to pop is empty.
//        - The caller thread of pop() is not equal to the _expectConsumer. This is to enforce
//          the ordering.
//
// Future improvement:
//   1. Fault tolerant: Right now, if one of the worker dies, the Connector will not work
//      properly.
template <class T>
class Connector {
 public:
  // Name: Constructor
  // Description: Initializing private members with the given input arguments.
  //              expect_consumer_ and pop_from_ is initialized to 0 as part of
  //              our requirements. We instantiate nProducers number of internal
  //              queues so that each producer thread can push to its queue without
  //              any sync overhead.
  // Constructor of Connector
  // Initializing private members with the given input arguments.
  // _expectConsumer and _popFrom is initialized to 0 as part of
  // our requirements. We instantiate nProducers number of internal
  // queues so that each producer thread can push to its queue without
  // any sync overhead.
  // @param n_producers The number of threads producing data into this DbConnector.
  // @param n_consumers The number of thread consuming data from this DbConnector.
  // @param queue_capacity The number of element (DataBuffer) for each queue.
  Connector(int32_t n_producers, int32_t n_consumers, int32_t queue_capacity)
      : num_producers_(n_producers), num_consumers_(n_consumers) {
    MS_LOG(DEBUG) << "A connector is created with " << n_producers << " producers and " << n_consumers << " consumers.";
    my_name_ = Services::GetUniqueID();
    // We require the consumers to have ids sequentially from 0 to the num_consumers_-1,
    // Otherwise a ordered list of consumer ids have to be passed here. (not implemented yet)
    expect_consumer_ = 0;

    // Roundrobin pop starts from index 0 of the queues_.
    pop_from_ = 0;

    // Initialize the queues_ to have num_producers_ number of queues.
    // Each queue is a blocking queue and has the same queue_capacity.
    queues_.Init(num_producers_, queue_capacity);
  }

  // Destructor of Connector
  virtual ~Connector() = default;

  // Get an element from the Connector.
  // @not Call to pop() can block the caller thread, see the blocking condition at the top of this file.
  // @param worker_id The id of a worker thread calling this method.
  // @param result The address of an object where the popped element will be placed.
  virtual Status Pop(int32_t worker_id,  // The worker-id of the caller. See the requirement at the top of this file.
                     T *result) noexcept {
    {
      DS_ASSERT(worker_id < num_consumers_);
      std::unique_lock<std::mutex> lk(m_);
      RETURN_IF_NOT_OK(cv_.Wait(&lk, [this, worker_id]() { return expect_consumer_ == worker_id; }));
      RETURN_IF_NOT_OK(queues_[pop_from_]->PopFront(result));
      pop_from_ = (pop_from_ + 1) % num_producers_;
      out_buffers_count_++;
      expect_consumer_ = (expect_consumer_ + 1) % num_consumers_;
    }

    cv_.NotifyAll();
    return Status::OK();
  }

  // Add an element into the DbConnector without the overhead of synchronization.
  // It may block when the internal queue is full.
  // The element passed to this function will be copied into the internal queue.
  // @param worker_id The id of a worker thread calling this method.
  // @param el A const lvalue element to be passed/added/pushed.
  Status Push(int32_t worker_id, const T &el) noexcept {
    DS_ASSERT(worker_id < static_cast<int32_t>(queues_.size()));
    DS_ASSERT(queues_[worker_id] != nullptr);
    return (queues_[worker_id]->Add(el));
  }

  auto out_buffers_count() const { return out_buffers_count_.load(); }

  // Add an element into the DbConnector without the overhead of synchronization.
  // It may block when the internal queue is full.
  // The element passed to this function will be forwarded into the internal queue.
  // @param worker_id The id of a worker thread calling this method.
  // @param el An element to be passed/added/pushed.
  virtual Status Push(int32_t worker_id, T &&el) noexcept {
    DS_ASSERT(worker_id < static_cast<int32_t>(queues_.size()));
    DS_ASSERT(queues_[worker_id] != nullptr);
    return (queues_[worker_id]->Add(std::forward<T>(el)));
  }

  // Resets the internal index tracking of the queue so that it can be used again with new inputs,
  // starting from the beginning.
  void Reset() {
    for (int i = 0; i < queues_.size(); ++i) {
      queues_[i]->ResetQue();
    }
    expect_consumer_ = 0;
    pop_from_ = 0;
    out_buffers_count_ = 0;
    MS_LOG(DEBUG) << "Connector counters reset.";
  }

  void Print(std::ostream &out, bool showAll) const {
    out << "\n--------- Connector ------------"
        << "\nConnector Name           : " << my_name_ << "\nNumber of consumers      : " << num_consumers_
        << "\nNumber of producers      : " << num_producers_ << "\n";
  }

  friend std::ostream &operator<<(std::ostream &out, const Connector &con) {
    con.print(out, false);
    return out;
  }

  // Get current size of connector.
  int32_t size() const {
    int32_t size = 0;
    for (int32_t i = 0; i < queues_.size(); ++i) {
      size += queues_[i]->size();
    }
    return size;
  }

  int32_t capacity() const {
    int32_t capacity = 0;
    for (int32_t i = 0; i < queues_.size(); ++i) {
      capacity += queues_[i]->capacity();
    }
    return capacity;
  }

  // Register the internal resources with Task group for interruption service.
  // @param vg
  // @return
  Status Register(TaskGroup *vg) {
    Status rc = queues_.Register(vg);
    if (rc.IsOk()) {
      rc = cv_.Register(vg->GetIntrpService());
    }
    return rc;
  }

 protected:
  std::string my_name_;

  // A list of Queues that are thread safe.
  QueueList<T> queues_;

  // The consumer that we allow to get the next data from pop()
  int32_t expect_consumer_;

  // The index to the queues_ where the next data should be popped.
  int32_t pop_from_;

  int32_t num_producers_;
  int32_t num_consumers_;

  // Used in the Pop(), when a thread call pop() but it is not the expect_consumer_.
  std::mutex m_;
  CondVar cv_;
  std::atomic<std::int64_t> out_buffers_count_ = 0;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_CONNECTOR_H_
