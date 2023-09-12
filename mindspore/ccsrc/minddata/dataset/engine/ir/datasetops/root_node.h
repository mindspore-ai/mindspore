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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_ROOT_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_ROOT_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class RootNode : public DatasetNode {
 public:
  /// \brief Constructor
  RootNode() : DatasetNode(), num_epochs_(0), step_(0), dataset_size_(-1) {}

  /// \brief Constructor
  explicit RootNode(std::shared_ptr<DatasetNode> child);

  /// \brief Destructor
  ~RootNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kRootNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) override;

  /// \brief Getter of number of epochs
  int32_t NumEpochs() const { return num_epochs_; }

  /// \brief Getter of number of epochs
  int64_t Step() const { return step_; }

  /// \brief Getter of number of steps in one epoch
  int64_t DatasetSize() const { return dataset_size_; }

  /// \brief Setter of number of epochs
  void SetStep(int64_t step) { step_ = step; }

  /// \brief Setter of number of epochs
  void SetNumEpochs(int32_t num_epochs) override { num_epochs_ = num_epochs; }

  /// \brief Setter of number of steps in one epoch
  void SetDatasetSize(int64_t dataset_size) { dataset_size_ = dataset_size; }

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

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
  int32_t num_epochs_;
  int64_t step_;  // to support reset
  int64_t dataset_size_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_ROOT_NODE_H_
