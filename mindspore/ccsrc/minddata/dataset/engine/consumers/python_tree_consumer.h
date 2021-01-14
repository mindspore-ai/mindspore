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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/consumers/tree_consumer.h"

namespace mindspore::dataset {

/// Consumer that iterates over the dataset and returns the rows one by one as a python list or a dict

class PythonIteratorConsumer : public IteratorConsumer {
 public:
  /// Constructor which will call the base class default constructor.
  /// \param num_epochs number of epochs. Default to -1 (infinite epochs).
  explicit PythonIteratorConsumer(int32_t num_epochs = -1) : IteratorConsumer(num_epochs) {}

  ~PythonIteratorConsumer() = default;
  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  Status GetNextAsList(py::list *out);

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  Status GetNextAsDict(py::dict *out);
};

class PythonBuildVocabConsumer : public BuildVocabConsumer {
 public:
  Status Start() override;
};

class PythonSaveToDisk : public SaveToDisk {
 public:
  PythonSaveToDisk(const std::string &datasetPath, int32_t numFiles, const std::string &datasetType);
  ~PythonSaveToDisk() = default;
  Status Save() override;
};

class PythonTreeGetters : public TreeGetters {
 public:
  Status GetRow(TensorRow *const r) override;
  ~PythonTreeGetters() = default;
};
class PythonDatasetSizeGetter : public DatasetSizeGetter {
 public:
  Status GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *r) override;
  ~PythonDatasetSizeGetter() = default;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_
