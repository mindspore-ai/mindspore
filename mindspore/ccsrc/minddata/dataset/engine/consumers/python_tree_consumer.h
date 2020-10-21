/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMER_PYTHON_TREE_CONSUMER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMER_PYTHON_TREE_CONSUMER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/consumers/tree_consumer.h"

namespace mindspore::dataset {

/// Consumer that iterates over the dataset and returns the rows one by one as a python list or a dict
class PythonIterator : public IteratorConsumer {
  /// Constructor
  /// \param num_epochs number of epochs. Default to -1 (infinite epochs).
  explicit PythonIterator(int32_t num_epochs = -1) : IteratorConsumer(num_epochs) {}

  /// Get the next row as a python dict
  /// \param[out] output python dict
  /// \return  Status error code
  Status GetNextAsMap(py::dict *output) {
    return Status(StatusCode::kNotImplementedYet, __LINE__, __FILE__, "Method is not implemented yet.");
  }
  /// Get the next row as a python dict
  /// \param[out] output python dict
  /// \return  Status error code
  Status GetNextAsList(py::list *output) {
    return Status(StatusCode::kNotImplementedYet, __LINE__, __FILE__, "Method is not implemented yet.");
  }
};

}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMER_PYTHON_TREE_CONSUMER_H_
