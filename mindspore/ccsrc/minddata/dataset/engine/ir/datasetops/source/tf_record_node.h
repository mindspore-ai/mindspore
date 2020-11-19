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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TF_RECORD_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TF_RECORD_NODE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

/// \class TFRecordNode
/// \brief A Dataset derived class to represent TFRecord dataset
class TFRecordNode : public NonMappableSourceNode {
 public:
  /// \brief Constructor
  /// \note Parameter 'schema' is the path to the schema file
  TFRecordNode(const std::vector<std::string> &dataset_files, std::string schema,
               const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
               int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache)
      : NonMappableSourceNode(std::move(cache)),
        dataset_files_(dataset_files),
        schema_path_(schema),
        columns_list_(columns_list),
        num_samples_(num_samples),
        shuffle_(shuffle),
        num_shards_(num_shards),
        shard_id_(shard_id),
        shard_equal_rows_(shard_equal_rows) {}

  /// \brief Constructor
  /// \note Parameter 'schema' is shared pointer to Schema object
  TFRecordNode(const std::vector<std::string> &dataset_files, std::shared_ptr<SchemaObj> schema,
               const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
               int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache)
      : NonMappableSourceNode(std::move(cache)),
        dataset_files_(dataset_files),
        schema_obj_(schema),
        columns_list_(columns_list),
        num_samples_(num_samples),
        shuffle_(shuffle),
        num_shards_(num_shards),
        shard_id_(shard_id),
        shard_equal_rows_(shard_equal_rows) {}

  /// \brief Destructor
  ~TFRecordNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kTFRecordNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

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
  std::vector<std::string> dataset_files_;
  std::string schema_path_;  // schema_path_ path to schema file. It is set when type of schema parameter is string
  std::shared_ptr<SchemaObj> schema_obj_;  // schema_obj_ schema object.
  std::vector<std::string> columns_list_;
  int64_t num_samples_;
  ShuffleMode shuffle_;
  int32_t num_shards_;
  int32_t shard_id_;
  bool shard_equal_rows_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TF_RECORD_NODE_H_
