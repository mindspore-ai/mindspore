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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/include/samplers.h"

namespace mindspore {
namespace dataset {

class RandomNode : public NonMappableSourceNode {
 public:
  // Some constants to provide limits to random generation.
  static constexpr int32_t kMaxNumColumns = 4;
  static constexpr int32_t kMaxRank = 4;
  static constexpr int32_t kMaxDimValue = 32;

  /// \brief Constructor
  RandomNode(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema, const std::vector<std::string> &columns_list,
             std::shared_ptr<DatasetCache> cache)
      : NonMappableSourceNode(std::move(cache)),
        total_rows_(total_rows),
        schema_path_(""),
        schema_(std::move(schema)),
        columns_list_(columns_list) {}

  /// \brief Constructor
  RandomNode(const int32_t &total_rows, std::string schema_path, const std::vector<std::string> &columns_list,
             std::shared_ptr<DatasetCache> cache)
      : NonMappableSourceNode(std::move(cache)),
        total_rows_(total_rows),
        schema_path_(schema_path),
        schema_(nullptr),
        columns_list_(columns_list) {}

  /// \brief Destructor
  ~RandomNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kRandomNode; }

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
  int32_t TotalRows() const { return total_rows_; }
  const std::string &SchemaPath() const { return schema_path_; }
  const std::shared_ptr<SchemaObj> &GetSchema() const { return schema_; }
  const std::vector<std::string> &ColumnsList() const { return columns_list_; }
  const std::mt19937 &RandGen() const { return rand_gen_; }
  const std::unique_ptr<DataSchema> &GetDataSchema() const { return data_schema_; }

  /// \brief RandomDataset by itself is a non-mappable dataset that does not support sampling.
  ///     However, if a cache operator is injected at some other place higher in the tree, that cache can
  ///     inherit this sampler from the leaf, providing sampling support from the caching layer.
  ///     That is why we setup the sampler for a leaf node that does not use sampling.
  /// \param[in] sampler The sampler to setup
  /// \return Status of the function
  Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) override;

  /// \brief Random node will always produce the full set of data into the cache
  /// \return Status of the function
  Status MakeSimpleProducer() override { return Status::OK(); }

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;

 private:
  /// \brief A quick inline for producing a random number between (and including) min/max
  /// \param[in] min minimum number that can be generated.
  /// \param[in] max maximum number that can be generated.
  /// \return The generated random number
  int32_t GenRandomInt(int32_t min, int32_t max);

  int32_t total_rows_;
  std::string schema_path_;
  std::shared_ptr<SchemaObj> schema_;
  std::vector<std::string> columns_list_;
  std::mt19937 rand_gen_;
  std::unique_ptr<DataSchema> data_schema_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_
