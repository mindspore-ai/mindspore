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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_GENERATOR_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_GENERATOR_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \class GeneratorNode
/// \brief A Dataset derived class to represent GeneratorNode dataset
class GeneratorNode : public MappableSourceNode {
 public:
  /// \brief Constructor
  GeneratorNode(py::function generator_function, const std::vector<std::string> &column_names,
                const std::vector<DataType> &column_types);

  /// \brief Constructor
  GeneratorNode(py::function generator_function, const std::shared_ptr<SchemaObj> &schema);

  /// \brief Destructor
  ~GeneratorNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kGeneratorNode; }

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

  /// \brief Get the shard id of node, is always 0 because generator_node doesn't support sharding
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

  /// \brief Setter for DatasetSize in GeneratorNode
  /// \param[in] sz dataset size to set
  /// \return void
  void SetGeneratorDatasetSize(int64_t sz) { dataset_size_ = sz; }

  bool IsSizeDefined() override { return false; }

  /// \brief Record the vector of Repeat/EpochCtrl nodes that are ancestors of this node
  /// \param[in] the ancestor node
  /// \return Status of the function
  Status AddResetAncestor(const std::shared_ptr<RepeatNode> &src) {
    CHECK_FAIL_RETURN_UNEXPECTED(reset_ancestor_ == nullptr, "Internal error: Overwriting an existing value");
    reset_ancestor_ = src;
    return Status::OK();
  }

 private:
  py::function generator_function_;
  std::vector<std::string> column_names_;
  std::vector<DataType> column_types_;
  std::shared_ptr<SchemaObj> schema_;
  std::shared_ptr<RepeatNode> reset_ancestor_;  // updated its immediate Repeat/EpochCtrl ancestor in GeneratorNodePass

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
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_GENERATOR_NODE_H_
