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
  friend class CacheValidationPass;

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
        shard_equal_rows_(shard_equal_rows) {
    // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User
    // discretion is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the
    // num_shards_ isn't 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return
    // num_shards. Once PreBuildSampler is phased out, this can be cleaned up.
    GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
  }

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
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

  /// \brief Base-class override for GetDatasetSize
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset
  /// \return Status of the function
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

  /// \brief Get the file list of the specific shard ID
  /// \param[out] shard_filenames the list of filenames for that specific shard ID
  /// \return Status of the function
  Status GetShardFileList(std::vector<std::string> *shard_filenames);

  /// \brief Getter functions
  const std::vector<std::string> &DatasetFiles() const { return dataset_files_; }
  const std::string &SchemaPath() const { return schema_path_; }
  const std::shared_ptr<SchemaObj> &GetSchemaObj() const { return schema_obj_; }
  const std::vector<std::string> &ColumnsList() const { return columns_list_; }
  int64_t NumSamples() const { return num_samples_; }
  ShuffleMode Shuffle() const { return shuffle_; }
  int32_t NumShards() const { return num_shards_; }
  bool ShardEqualRows() const { return shard_equal_rows_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief TFRecord by itself is a non-mappable dataset that does not support sampling.
  ///     However, if a cache operator is injected at some other place higher in the tree, that cache can
  ///     inherit this sampler from the leaf, providing sampling support from the caching layer.
  ///     That is why we setup the sampler for a leaf node that does not use sampling.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \param[in] sampler The sampler to setup
  /// \return Status of the function
  Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) override;

  /// \brief If a cache has been added into the ascendant tree over this TFRecord node, then the cache will be executing
  ///     a sampler for fetching the data.  As such, any options in the TFRecord node need to be reset to its defaults
  ///     so that this TFRecord node will produce the full set of data into the cache.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \return Status of the function
  Status MakeSimpleProducer() override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *p, bool *const modified) override;

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
