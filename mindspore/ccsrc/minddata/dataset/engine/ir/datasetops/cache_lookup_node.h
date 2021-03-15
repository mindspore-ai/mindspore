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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_CACHE_LOOKUP_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_CACHE_LOOKUP_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
class CacheLookupNode : public DatasetNode, public SamplerObj {
 public:
  /// \brief Constructor
  CacheLookupNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<SamplerObj> sampler,
                  std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~CacheLookupNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kCacheLookupNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to convert a SamplerObj class into a runtime sampler object
  /// \param[out] out Shared pointer to the newly created Sampler
  /// \return The Status code of the function. It returns OK status if sampler is created successfully.
  Status SamplerBuild(std::shared_ptr<SamplerRT> *const out) override;

  /// \brief a base class override function to copy a SamplerObj class
  /// \return Shared pointers to the newly copied SamplerObj
  std::shared_ptr<SamplerObj> SamplerCopy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) override;

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
  std::shared_ptr<SamplerObj> sampler_;
  std::shared_ptr<DatasetOp> lookup_op_;
  std::shared_ptr<CacheLookupNode> lookup_node_copy_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_CACHE_LOOKUP_NODE_H_
