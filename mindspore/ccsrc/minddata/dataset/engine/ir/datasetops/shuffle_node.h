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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SHUFFLE_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SHUFFLE_NODE_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
class ShuffleNode : public DatasetNode {
 public:
  ShuffleNode(std::shared_ptr<DatasetNode> child, int32_t shuffle_size, bool reset_every_epoch);

  ~ShuffleNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kShuffleNode; }

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

  Status ValidateParams() override;

  /// \brief Getter functions
  int32_t ShuffleSize() const { return shuffle_size_; }
  uint32_t ShuffleSeed() const { return shuffle_seed_; }
  bool ResetEveryEpoch() const { return reset_every_epoch_; }

  /// \brief Setter function for shuffle_seed_
  /// \param[in] shuffle_seed The shuffle seed value to be set
  void SetShuffleSeed(uint32_t shuffle_seed) { shuffle_seed_ = shuffle_seed; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief Function for read dataset operation from json
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] ds dataset node constructed
  /// \param[out] result Deserialized dataset after the operation
  /// \return Status The status code returned
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                          std::shared_ptr<DatasetNode> *result);

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
  int32_t shuffle_size_;
  uint32_t shuffle_seed_;
  bool reset_every_epoch_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SHUFFLE_NODE_H_
