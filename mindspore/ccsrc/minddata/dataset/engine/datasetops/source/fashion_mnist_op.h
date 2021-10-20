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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FASHION_MNIST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FASHION_MNIST_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/mnist_op.h"

namespace mindspore {
namespace dataset {
/// \brief Forward declares.
template <typename T>
class Queue;

class FashionMnistOp : public MnistOp {
 public:
  /// \brief Constructor.
  /// \param[in] usage Usage of this dataset, can be 'train', 'test' or 'all'.
  /// \param[in] num_workers Number of workers reading images in parallel.
  /// \param[in] folder_path Dir directory of fashionmnist.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema The schema of the fashionmnist dataset.
  /// \param[in] Sampler Tells FashionMnistOp what to read.
  FashionMnistOp(const std::string &usage, int32_t num_workers, const std::string &folder_path, int32_t queue_size,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~FashionMnistOp() = default;

  /// \brief Function to count the number of samples in the Fashion-MNIST dataset.
  /// \param[in] dir Path to the Fashion-MNIST directory.
  /// \param[in] usage Usage of this dataset, can be 'train', 'test' or 'all'.
  /// \param[in] count Output arg that will hold the minimum of the actual dataset size and numSamples.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "FashionMnistOp"; }

  /// \brief Dataset name getter.
  /// \param[in] upper Whether to get upper name.
  /// \return Dataset name of the current Op.
  std::string DatasetName(bool upper = false) const override { return upper ? "FashionMnist" : "fashion mnist"; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FASHION_MNIST_OP_H_
