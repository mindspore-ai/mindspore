/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_EN_WIK9_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_EN_WIK9_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
/// \class EnWik9Node.
/// \brief A Dataset derived class to represent EnWik9 dataset.
class EnWik9Node : public NonMappableSourceNode {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_dir The directory of dataset.
  /// \param[in] num_samples The number of samples that users want to get.
  /// \param[in] shuffle Decide the dataset shuffle pattern.
  /// \param[in] num_shards The number of shards that users want to part.
  /// \param[in] shard_id The id of shard.
  /// \param[in] cache Tensor cache to use.
  EnWik9Node(const std::string &dataset_dir, int32_t num_samples, ShuffleMode shuffle, int32_t num_shards,
             int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor.
  ~EnWik9Node() override = default;

  /// \brief Node name getter.
  /// \return Name of the current node.
  std::string Name() const override { return kEnWik9Node; }

  /// \brief Print the description.
  /// \param[out] out The output stream to write output to.
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object.
  /// \return A shared pointer to the new copy.
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class.
  /// \param[in] node_ops A vector containing shared pointer to the Dataset Ops that this object will create.
  /// \return Status Status::OK() if build successfully.
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation.
  /// \return Status Status::OK() if all the parameters are valid.
  Status ValidateParams() override;

  /// \brief Get the shard id of node.
  /// \param[in] shard_id Id of this shard.
  /// \return Status Status::OK() if get shard id successfully.
  Status GetShardId(int32_t *shard_id) override;

  /// \brief Base-class override for GetDatasetSize.
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter.
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset.
  /// \return Status of the function.
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

  /// \brief Getter functions.
  /// \return Directory of dataset.
  const std::string &DatasetDir() const { return dataset_dir_; }

  // \brief Getter functions.
  /// \return The number of samples.
  int32_t NumSamples() const { return num_samples_; }

  // \brief Getter functions.
  /// \return The number of shards.
  int32_t NumShards() const { return num_shards_; }

  // \brief Getter functions.
  /// \return Id of shard.
  int32_t ShardId() const { return shard_id_; }

  // \brief Getter functions.
  /// \return Shuffle pattern.
  ShuffleMode Shuffle() const { return shuffle_; }

  /// \brief Get the arguments of node.
  /// \param[out] out_json JSON string of all attributes.
  /// \return Status of the function.
  Status to_json(nlohmann::json *out_json) override;

  /// \brief EnWik9 by itself is a non-mappable dataset that does not support sampling.
  ///     However, if a cache operator is injected at some other place higher in the tree, that cache can
  ///     inherit this sampler from the leaf, providing sampling support from the caching layer.
  ///     That is why we setup the sampler for a leaf node that does not use sampling.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \param[in] sampler The sampler to setup.
  /// \return Status of the function.
  Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) override;

  /// \brief If a cache has been added into the ascendant tree over this EnWik9 node, then the cache will be executing.
  ///     a sampler for fetching the data. As such, any options in the EnWik9 node need to be reset to its defaults
  ///     so that this EnWik9 node will produce the full set of data into the cache.
  ///     Note: This function is common among NonMappableSourceNode and should be promoted to its parent class.
  /// \return Status of the function.
  Status MakeSimpleProducer() override;

  /// \brief Change file's directory into file's path, and put it into a list.
  /// \param[in] dataset_dir Directory of enwik9 dataset.
  /// \return A list of read file names.
  void DirToPath(const std::string &dataset_dir);

 private:
  std::string dataset_dir_;                        // dataset of file.
  int32_t num_samples_;                            // the number of samples.
  int32_t num_shards_;                             // the number of shards.
  int32_t shard_id_;                               // the id of shard.
  ShuffleMode shuffle_;                            // a object of ShuffleMode, which belongs to num.
  std::vector<std::string> src_target_file_list_;  // file list;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_EN_WIK9_NODE_H_
