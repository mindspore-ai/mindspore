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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_IMAGE_FOLDER_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_IMAGE_FOLDER_NODE_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindspore/ccsrc/minddata/dataset/engine/ir/cache/dataset_cache.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

/// \class ImageFolderNode
/// \brief A Dataset derived class to represent ImageFolder dataset
class ImageFolderNode : public MappableSourceNode {
 public:
  /// \brief Constructor
  ImageFolderNode(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler, bool recursive,
                  std::set<std::string> extensions, std::map<std::string, int32_t> class_indexing,
                  std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~ImageFolderNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kImageFolderNode; }

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
  const std::string &DatasetDir() const { return dataset_dir_; }
  bool Decode() const { return decode_; }
  bool Recursive() const { return recursive_; }
  const std::map<std::string, int32_t> &ClassIndexing() const { return class_indexing_; }
  const std::set<std::string> &Exts() const { return exts_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief Sampler getter
  /// \return SamplerObj of the current node
  std::shared_ptr<SamplerObj> Sampler() override { return sampler_; }

  /// \brief Sampler setter
  void SetSampler(std::shared_ptr<SamplerObj> sampler) override { sampler_ = sampler; }

 private:
  std::string dataset_dir_;
  bool decode_;
  bool recursive_;
  std::shared_ptr<SamplerObj> sampler_;
  std::map<std::string, int32_t> class_indexing_;
  std::set<std::string> exts_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_IMAGE_FOLDER_NODE_H_
