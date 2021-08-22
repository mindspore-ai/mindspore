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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

/// \class CLUENode
/// \brief A Dataset derived class to represent CLUE dataset
class CLUENode : public NonMappableSourceNode {
 public:
  /// \brief Constructor
  CLUENode(const std::vector<std::string> dataset_files, std::string task, std::string usage, int64_t num_samples,
           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~CLUENode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kCLUENode; }

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

  /// \brief Getter functions
  const std::vector<std::string> &DatasetFiles() const { return dataset_files_; }
  const std::string &Task() const { return task_; }
  const std::string &Usage() const { return usage_; }
  int64_t NumSamples() const { return num_samples_; }
  ShuffleMode Shuffle() const { return shuffle_; }
  int32_t NumShards() const { return num_shards_; }
  int32_t ShardId() const { return shard_id_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief CLUE by itself is a non-mappable dataset that does not support sampling.
  ///     However, if a cache operator is injected at some other place higher in the tree, that cache can
  ///     inherit this sampler from the leaf, providing sampling support from the caching layer.
  ///     That is why we setup the sampler for a leaf node that does not use sampling.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \param[in] sampler The sampler to setup
  /// \return Status of the function
  Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) override;

  /// \brief If a cache has been added into the ascendant tree over this clue node, then the cache will be executing
  ///     a sampler for fetching the data.  As such, any options in the clue node need to be reset to its defaults so
  ///     that this clue node will produce the full set of data into the cache.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \return Status of the function
  Status MakeSimpleProducer() override;

 private:
  /// \brief Split string based on a character delimiter
  /// \return A string vector
  std::vector<std::string> split(const std::string &s, char delim);

  /// \brief Generate a key map for AFQMC or CMNLI task according to usage
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMapForAFQMCOrCMNLITask();

  /// \brief Generate a key map for CSL task according to usage
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMapForCSLTask();

  /// \brief Generate a key map for IFLYTEK task according to usage
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMapForIFLYTEKTask();

  /// \brief Generate a key map for TNEWS task according to usage
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMapForTNEWSTask();

  /// \brief Generate a key map for WSC task according to usage
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMapForWSCTask();

  /// \brief Generate a key map to be used in Build() according to usage and task
  /// \return The generated key map
  std::map<std::string, std::string> CreateKeyMap();

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
