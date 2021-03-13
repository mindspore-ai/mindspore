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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DB_CONNECTOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DB_CONNECTOR_H_

#include <memory>
#include <utility>
#include "minddata/dataset/engine/connector.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {
// DbConnector is a derived class from Connector with added logic to handle EOE and EOF.
// The Connector class itself is responsible to ensure deterministic order on every run.
class DbConnector : public Connector<std::unique_ptr<DataBuffer>> {
 public:
  // Constructor of DbConnector
  // @note DbConnector will create internal N number of blocking queues, where N = nProducers.
  //     See Connector.h for more details.
  // @param n_producers The number of threads producing data into this DbConnector.
  // @param n_consumers The number of thread consuming data from this DbConnector.
  // @param queue_capacity The number of element (DataBuffer) for each internal queue.
  DbConnector(int32_t n_producers, int32_t n_consumers, int32_t queue_capacity)
      : Connector<std::unique_ptr<DataBuffer>>(n_producers, n_consumers, queue_capacity), end_of_file_(false) {}

  // Destructor of DbConnector
  ~DbConnector() = default;

  // Add a unique_ptr<DataBuffer> into the DbConnector.
  // @note The caller of this add method should use std::move to pass the ownership to DbConnector.
  // @param worker_id The id of a worker thread calling this method.
  // @param el A rvalue reference to an element to be passed/added/pushed.
  Status Add(int32_t worker_id, std::unique_ptr<DataBuffer> &&el) noexcept {
    return (Connector<std::unique_ptr<DataBuffer>>::Push(worker_id, std::move(el)));
  }

  // Get a unique_ptr<DataBuffer> from the DbConnector.
  // @note After the first EOF Buffer is encountered, subsequent pop()s will return EOF Buffer.
  // This will provide/propagate the EOF to all consumer threads of this Connector.
  // Thus, When the num_consumers < num_producers, there will be extra EOF messages in some of the internal queues
  // and reset() must be called before reusing DbConnector.
  // @param worker_id The id of a worker thread calling this method.
  // @param result The address of a unique_ptr<DataBuffer> where the popped element will be placed.
  // @param retry_if_eoe A flag to allow the same thread invoke pop() again if the current pop returns eoe buffer.
  Status PopWithRetry(int32_t worker_id, std::unique_ptr<DataBuffer> *result, bool retry_if_eoe = false) noexcept {
    if (result == nullptr) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                    "[ERROR] nullptr detected when getting data from db connector");
    } else {
      std::unique_lock<std::mutex> lk(m_);
      RETURN_IF_NOT_OK(cv_.Wait(&lk, [this, worker_id]() { return (expect_consumer_ == worker_id) || end_of_file_; }));
      // Once an EOF message is encountered this flag will be set and we can return early.
      if (end_of_file_) {
        *result = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
      } else {
        RETURN_IF_NOT_OK(queues_[pop_from_]->PopFront(result));
        if (*result == nullptr) {
          return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                        "[ERROR] nullptr detected when getting data from db connector");
        }
        // Setting the internal flag once the first EOF is encountered.
        if ((*result)->eof()) {
          end_of_file_ = true;
        }
        pop_from_ = (pop_from_ + 1) % num_producers_;
      }
      // Do not increment expect_consumer_ when result is eoe and retry_if_eoe is set.
      if (!((*result)->eoe() && retry_if_eoe)) {
        expect_consumer_ = (expect_consumer_ + 1) % num_consumers_;
      }
    }
    out_buffers_count_++;
    cv_.NotifyAll();
    return Status::OK();
  }

 private:
  // A flag to indicate the end of stream has been encountered.
  bool end_of_file_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DB_CONNECTOR_H_
