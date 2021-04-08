/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PULL_BASED_TREE_CONSUMER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PULL_BASED_TREE_CONSUMER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstddef>
#include "minddata/dataset/engine/tree_adapter_lite.h"

namespace mindspore::dataset {

class TreeAdapterLite;
class TensorRow;

/// Consumer that iterates over the dataset and returns the rows one by one as a in a pull based fashion
class PullBasedIteratorConsumer {
 public:
  /// Constructor
  PullBasedIteratorConsumer();

  ~PullBasedIteratorConsumer() = default;

  Status Init(std::shared_ptr<DatasetNode> root);

  /// \brief Returns the next row in a vector format
  /// \note This is currently a placeholder function
  /// \param[in] num_rows the number of rows that we want to get
  /// \param[out] out std::vector of TensorRows
  /// \return Status error code
  std::vector<TensorRow> GetRows(int64_t num_rows);

  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  Status GetNextAsVector(std::vector<TensorPtr> *const out);

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  Status GetNextAsMap(std::unordered_map<std::string, TensorPtr> *out);

  /// Returns the next row in as a vector
  /// \param[out] out std::vector of pairs of string to Tensor
  /// \return Status error code
  Status GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *vec);

 private:
  std::unique_ptr<TreeAdapterLite> tree_adapter_lite_;
  std::vector<std::pair<std::string, int32_t>> column_order_;  // key: column name, val: column id
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PULL_BASED_TREE_CONSUMER_H_
