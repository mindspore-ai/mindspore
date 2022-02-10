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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SOGOU_NEWS_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SOGOU_NEWS_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sogou_news_op.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
/// \class SogouNewsNode
/// \brief A Node derived class to represent SogouNews Node.
class SogouNewsNode : public NonMappableSourceNode {
 public:
  /// \brief Constructor of SogouNewsNode.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SogouNews, can be "train", "test" or "all" data.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  SogouNewsNode(const std::string &dataset_dir, const std::string &usage, int64_t num_samples, ShuffleMode shuffle,
                int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor.
  ~SogouNewsNode() override = default;

  /// \brief Node name getter.
  /// \return Name of the current node.
  std::string Name() const override { return kSogouNewsNode; }

  /// \brief Print the description.
  /// \param[out] out The output stream to write output to.
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object.
  /// \return A shared pointer to the new copy.
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief A base class override function to create the required runtime dataset op objects for this class.
  /// \param node_ops A vector containing shared pointer to the Dataset Ops that this object will create.
  /// \return Status Status::OK() if build successfully.
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation.
  /// \return Status Status::OK() if all the parameters are valid.
  Status ValidateParams() override;

  /// \brief Get the shard id of node.
  /// \param[in] shard_id The shard id.
  /// \return Status Status::OK() if get shard id successfully.
  Status GetShardId(int32_t *shard_id) override;

  /// \brief Base-class override for GetDatasetSize.
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter.
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting.
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size The size of the dataset.
  /// \return Status of the function.
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

  /// \brief Getter functions.
  const std::string &DatasetDir() const { return dataset_dir_; }
  const std::string &Usage() const { return usage_; }
  int64_t NumSamples() const { return num_samples_; }
  ShuffleMode Shuffle() const { return shuffle_; }
  int32_t NumShards() const { return num_shards_; }
  int32_t ShardId() const { return shard_id_; }

  /// \brief Get the arguments of node.
  /// \param[out] out_json JSON string of all attributes.
  /// \return Status of the function.
  Status to_json(nlohmann::json *out_json) override;

  /// \brief SogouNews by itself is a non-mappable dataset that does not support sampling.
  ///     However, if a cache operator is injected at some other place higher in the tree, that cache can
  ///     inherit this sampler from the leaf, providing sampling support from the caching layer.
  ///     That is why we setup the sampler for a leaf node that does not use sampling.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \param[in] sampler The sampler to setup.
  /// \return Status of the function.
  Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) override;

  /// \brief If a cache has been added into the ascendant tree over this clue node, then the cache will be executing
  ///     a sampler for fetching the data. As such, any options in the clue node need to be reset to its defaults so.
  ///     that this clue node will produce the full set of data into the cache.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \return Status of the function.
  Status MakeSimpleProducer() override;

  /// \brief Generate a list of read file names according to usage.
  /// \param[in] usage Part of dataset of SogouNews.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \return std::vector<std::string> A list of read file names.
  std::vector<std::string> WalkAllFiles(const std::string &usage, const std::string &dataset_dir);

 private:
  std::string dataset_dir_;
  std::string usage_;
  std::vector<std::shared_ptr<CsvBase>> column_defaults_;
  std::vector<std::string> column_names_;
  int64_t num_samples_;
  ShuffleMode shuffle_;
  int32_t num_shards_;
  int32_t shard_id_;
  std::vector<std::string> sogou_news_files_list_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SOGOU_NEWS_NODE_H_
