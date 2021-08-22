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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_REPEAT_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_REPEAT_NODE_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class RepeatOp;

class RepeatNode : public DatasetNode {
  // Allow GeneratorNode to access internal members
  friend class GeneratorNode;

 public:
  /// \brief Constructor
  RepeatNode() : op_(nullptr), reset_ancestor_(nullptr), repeat_count_(-1) {}

  /// \brief Constructor
  RepeatNode(std::shared_ptr<DatasetNode> child, int32_t count);

  /// \brief Destructor
  ~RepeatNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kRepeatNode; }

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

  /// \brief Getter
  /// \return Number of cycles to repeat the execution
  const int32_t Count() const { return repeat_count_; }

  /// \brief Base-class override for GetDatasetSize
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset
  /// \return Status of the function
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

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

  /// \brief Record the Repeat/EpochCtrl node that is the closest ancestor of this node
  /// \param[in] the ancestor node
  /// \return Status of the function
  Status AddResetAncestor(const std::shared_ptr<RepeatNode> &src) {
    /*
     * This check is to ensure we don't overwrite an existing value of its ancestor.
     * It is okay to assign to the same value more than once in RepeatNode (but not in GeneratorNode).
     * Consider the following scenario
     *       EpochCtrl(-1)
     *           |
     *        Repeat
     *           |
     *        Concat
     *        /    \
     *  GenData1  GenData2
     *
     * We will record the ancestor relationship of (Repeat, EpochCtrl) twice, one at Visit(GenData1), the other at
     * Vist(GenData2).
     */
    CHECK_FAIL_RETURN_UNEXPECTED(reset_ancestor_ == nullptr || reset_ancestor_ == src,
                                 "Internal error: Overwriting an existing value");
    reset_ancestor_ = src;
    return Status::OK();
  }

  /// \brief Getter functions
  int32_t RepeatCount() const { return repeat_count_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 protected:
  std::shared_ptr<RepeatOp> op_;                // keep its corresponding run-time op of EpochCtrlNode and RepeatNode
  std::shared_ptr<RepeatNode> reset_ancestor_;  // updated its immediate Repeat/EpochCtrl ancestor in GeneratorNodePass
  int32_t repeat_count_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_REPEAT_NODE_H_
