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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_CACHE_PASS_H_
#define DATASET_ENGINE_OPT_PASS_PRE_CACHE_PASS_H_

#include <memory>
#include <string>
#include <utility>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class CacheTransformPass;

/// \class CachePass cache_pass.h
/// \brief This is a NodePass who's job is to identify and set up the nodes that will be involved in a cache
///     transformation. It works in conjunction with the CacheTransformPass
class CachePass : public NodePass {
 public:
  /// \brief Constructor
  /// \param[in] transform_pass Raw pointer back to controlling tree pass
  explicit CachePass(CacheTransformPass *transform_pass);

  /// \brief Identifies the subtree below this node as a cached descendant tree.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) override;

  /// \brief Resets the tracking of the cache within the tree and assigns the operators that will be involved in a cache
  ///     transformation
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<RandomDataOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<MnistOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CocoOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified) override;

  /// \brief Perform leaf node cache tranform identifications
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) override;

 private:
  /// \brief Common code for mappable leaf setup.
  /// \param[in] node The leaf node performing setup work.
  /// \return Status The error code return
  Status MappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op);

  /// \brief Common code for non-mappable leaf setup.
  /// \param[in] node The leaf node performing setup work.
  /// \return Status The error code return
  Status NonMappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op);

  bool is_caching_;
  std::shared_ptr<DatasetOp> leaf_op_;
  std::shared_ptr<Sampler> sampler_;
  CacheTransformPass *transform_pass_;  // Back pointer to the owning transform pass
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_CACHE_PASS_
