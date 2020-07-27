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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_

#include <memory>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

class CacheClient;

/// \class CacheTransformPass cache_transform_pass.h
/// \brief This is a tree pass that will invoke a tree transformation to inject the correct operators for caching
///     operations
class CacheTransformPass : public TreePass {
  /// \class CachePass
  /// \brief This is a NodePass who's job is to identify and set up the nodes that will be involved in a cache
  ///     transformation. It works in conjunction with the CacheTransformPass
  class CachePass : public NodePass {
   public:
    /// \brief Constructor
    /// \param[in] transform_pass Raw pointer back to controlling tree pass
    CachePass();

    /// \brief Destructor
    ~CachePass() = default;

    /// \brief Identifies the subtree below this node as a cached descendant tree.
    /// \param[in] node The node being visited
    /// \param[inout] modified Indicator if the node was changed at all
    /// \return Status The error code return
    Status PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) override;

    /// \brief Resets the tracking of the cache within the tree and assigns the operators that
    ///     will be involved in a cache transformation
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

#ifdef ENABLE_PYTHON
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
    Status RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) override;
#endif

    /// \brief Perform leaf node cache tranform identifications
    /// \param[in] node The node being visited
    /// \param[inout] modified Indicator if the node was changed at all
    /// \return Status The error code return
    Status RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) override;

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

    /// \brief Getter
    std::vector<std::pair<std::shared_ptr<DatasetOp>, std::shared_ptr<CacheOp>>> cache_pairs() { return cache_pairs_; }

   private:
    /// \brief Common code for mappable leaf setup.
    /// \param[in] node The leaf node performing setup work.
    /// \return Status The error code return
    Status MappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op);

    /// \brief Common code for non-mappable leaf setup.
    /// \param[in] node The leaf node performing setup work.
    /// \return Status The error code return
    Status NonMappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op);

    /// \brief Assigns the leaf and cache operators that are involved in a cache transformation
    /// \param[in] leaf_op The leaf operator involved in the cache transform
    /// \param[in] cache_op The cache operator involved in the cache transform
    void AddMappableCacheOperators(std::shared_ptr<DatasetOp> leaf_op, std::shared_ptr<CacheOp> cache_op);

    bool is_caching_;
    std::shared_ptr<DatasetOp> leaf_op_;
    std::shared_ptr<Sampler> sampler_;
    // The two operators that work together to establish the cache transform
    std::vector<std::pair<std::shared_ptr<DatasetOp>, std::shared_ptr<CacheOp>>> cache_pairs_;
  };

 public:
  /// \brief Constructor
  CacheTransformPass();

  /// \brief Destructor
  ~CacheTransformPass() = default;

  /// \brief Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
  /// \param[inout] tree The tree to operate on.
  /// \param[inout] Indicate of the tree was modified.
  /// \return Status The error code return
  Status RunOnTree(ExecutionTree *tree, bool *modified) override;

 private:
  /// \brief Helper function to execute the cache transformation.
  ///
  ///     Input:
  ///       Sampler
  ///         |
  ///       LeafOp --> OtherOps --> CacheOp
  ///
  ///     Transformed:
  ///       Sampler --> CacheLookupOp ---------------->
  ///                           |                       |
  ///                           |                       MergeOp
  ///                           |                       |
  ///                           LeafOp --> OtherOps -->
  ///
  /// \param[in] leaf_op The leaf node in the transform
  /// \param[in] cache_op The cache op in the transform (will get removed)
  /// \param[in] cache_client The cache client
  /// \return Status The error code return
  Status ExecuteCacheTransform(ExecutionTree *tree, std::shared_ptr<DatasetOp> leaf_op,
                               std::shared_ptr<DatasetOp> cache_op, std::shared_ptr<CacheClient> cache_client);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_
