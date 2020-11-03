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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CLUE_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CLUE_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

/// \class CLUENode
/// \brief A Dataset derived class to represent CLUE dataset
class CLUENode : public DatasetNode {
 public:
  /// \brief Constructor
  CLUENode(const std::vector<std::string> dataset_files, std::string task, std::string usage, int64_t num_samples,
           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~CLUENode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

 private:
  /// \brief Split string based on a character delimiter
  /// \return A string vector
  std::vector<std::string> split(const std::string &s, char delim);

  std::vector<std::string> dataset_files_;
  std::string task_;
  std::string usage_;
  int64_t num_samples_;
  ShuffleMode shuffle_;
  int32_t num_shards_;
  int32_t shard_id_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CLUE_NODE_H_
