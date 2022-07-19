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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPERATOR_CONNECTOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPERATOR_CONNECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/connector.h"

#include "minddata/dataset/include/dataset/constants.h"

namespace mindspore {
namespace dataset {

class OperatorConnector : public Queue<TensorRow> {
 public:
  /// Constructor of OperatorConnector
  /// \param queue_capacity The number of element (TensorRows) for the queue.
  explicit OperatorConnector(int32_t queue_capacity) : Queue<TensorRow>(queue_capacity), out_rows_count_(0) {}

  /// Destructor of -OperatorConnector
  ~OperatorConnector() = default;

  Status PopFront(TensorRow *row) override {
    out_rows_count_++;
    return Queue::PopFront(row);
  }
  Status SendEOE() noexcept {
    TensorRow eoe = TensorRow(TensorRow::kFlagEOE);
    return Add(std::move(eoe));
  }

  Status SendEOF() noexcept {
    TensorRow eof = TensorRow(TensorRow::kFlagEOF);
    return Add(std::move(eof));
  }
  auto out_rows_count() const { return out_rows_count_; }

 private:
  int64_t out_rows_count_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPERATOR_CONNECTOR_H_
