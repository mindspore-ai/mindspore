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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_JAGGED_CONNECTOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_JAGGED_CONNECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/connector.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {
class JaggedConnector : public Connector<std::unique_ptr<DataBuffer>> {
 public:
  JaggedConnector(int32_t num_producers, int32_t num_consumers, int32_t queue_capacity)
      : Connector<std::unique_ptr<DataBuffer>>(num_producers, num_consumers, queue_capacity) {
    for (int i = 0; i < num_producers; i++) {
      is_queue_finished_.push_back(false);
    }
  }

  ~JaggedConnector() = default;

  Status Add(int32_t worker_d, std::unique_ptr<DataBuffer> &&element) noexcept {
    return Connector<std::unique_ptr<DataBuffer>>::Push(worker_d, std::move(element));
  }

  Status Pop(int32_t worker_id, std::unique_ptr<DataBuffer> *result) noexcept override {
    {
      MS_ASSERT(worker_id < num_consumers_);
      std::unique_lock<std::mutex> lock(m_);
      RETURN_IF_NOT_OK(cv_.Wait(&lock, [this, worker_id]() { return expect_consumer_ == worker_id; }));
      if (is_queue_finished_[pop_from_]) {
        std::string errMsg = "ERROR: popping from a finished queue in JaggedConnector";
        RETURN_STATUS_UNEXPECTED(errMsg);
      }

      RETURN_IF_NOT_OK(queues_[pop_from_]->PopFront(result));
      if ((*result)->eoe()) {
        is_queue_finished_[pop_from_] = true;
      }

      for (int offset = 1; offset <= num_producers_; offset++) {
        int32_t nextQueueIndex = (pop_from_ + offset) % num_producers_;
        if (is_queue_finished_[nextQueueIndex] == false) {
          pop_from_ = nextQueueIndex;
          break;
        }
      }

      expect_consumer_ = (expect_consumer_ + 1) % num_consumers_;
    }

    cv_.NotifyAll();
    return Status::OK();
  }

  void DoReset() {
    for (int i = 0; i < is_queue_finished_.size(); i++) {
      is_queue_finished_[i] = false;
    }

    Connector<std::unique_ptr<DataBuffer>>::Reset();
  }

 private:
  std::vector<bool> is_queue_finished_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_JAGGED_CONNECTOR_H_
