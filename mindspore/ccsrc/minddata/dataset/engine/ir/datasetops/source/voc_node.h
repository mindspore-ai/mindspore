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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_VOC_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_VOC_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class VOCNode : public DatasetNode {
 public:
  /// \brief Constructor
  VOCNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
          const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler,
          std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~VOCNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

 private:
  const std::string kColumnImage = "image";
  const std::string kColumnTarget = "target";
  const std::string kColumnBbox = "bbox";
  const std::string kColumnLabel = "label";
  const std::string kColumnDifficult = "difficult";
  const std::string kColumnTruncate = "truncate";
  std::string dataset_dir_;
  std::string task_;
  std::string usage_;
  std::map<std::string, int32_t> class_index_;
  bool decode_;
  std::shared_ptr<SamplerObj> sampler_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_VOC_NODE_H_
