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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_

#include <memory>
#include <list>
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class GetterPass
/// \brief This is a tree pass that will remove nodes or clears the callback in MapOp
class GetterPass : public TreePass {
 public:
  enum GetterType { kDatasetSize = 1, kOutputShapeAndType = 2 };
  /// \brief Constructor
  explicit GetterPass(GetterType tp) : pass_(tp) {}

  /// \brief default copy Constructor
  explicit GetterPass(const GetterPass &) = default;

  /// \brief Destructor
  ~GetterPass() = default;

  Status RunOnTree(ExecutionTree *tree, bool *modified) override;

 private:
  /// \class GetterNodes, this is a nested class which is owned via composition by the outter class to identify nodes
  /// \brief This is a NodePass who's job is to identify which nodes should be removed.
  class GetterNodes : public NodePass {
   public:
    /// \brief Constructor
    explicit GetterNodes(GetterType tp) : type_(tp) {}

    ~GetterNodes() = default;

    Status RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) override;
    Status RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) override;
    Status RunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified) override { return Status::OK(); }
    Status RunOnNode(std::shared_ptr<SkipOp> node, bool *modified) override;
    Status RunOnNode(std::shared_ptr<TakeOp> node, bool *modified) override;
    Status RunOnNode(std::shared_ptr<MapOp> node, bool *modified) override;

#ifdef ENABLE_PYTHON
    Status RunOnNode(std::shared_ptr<FilterOp> node, bool *modified) override;
#endif

    GetterType type_;
    std::list<std::shared_ptr<DatasetOp>> nodes_to_clear_callback_;
    std::list<std::shared_ptr<DatasetOp>> nodes_to_remove_;
  };
  // outer class needs only to own the inner class object since it automatically has access to its private variables
  GetterNodes pass_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
