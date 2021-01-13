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

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

class CacheClient;

/// \class CacheTransformPass cache_transform_pass.h
/// \brief This is a tree pass that will invoke a tree transformation to inject the correct operators for caching
///     operations
class CacheTransformPass : public IRTreePass {
  /// \class CachePass
  /// \brief This is a NodePass who's job is to identify and set up the nodes that will be involved in a cache
  ///     transformation. It works in conjunction with the CacheTransformPass
  class CachePass : public IRNodePass {
   public:
    /// \brief Constructor
    /// \param[in] transform_pass Raw pointer back to controlling tree pass
    CachePass();

    /// \brief Destructor
    ~CachePass() = default;

    /// \brief Identifies the subtree below this node as a cached descendant tree.
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;

    /// \brief Resets the tracking of the cache within the tree and assigns the operators that
    ///     will be involved in a cache transformation
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

#ifndef ENABLE_ANDROID

    /// \brief Perform non-mappable leaf node cache transform identifications
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) override;
#endif

    /// \brief Perform non-mappable leaf node cache transform identifications
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RandomNode> node, bool *const modified) override;

    /// \brief Perform mappable leaf node cache transform identifications
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) override;

#ifdef ENABLE_PYTHON
    /// \brief Perform leaf node cache transform identifications
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) override;
#endif

#ifndef ENABLE_ANDROID
    /// \brief Perform leaf node cache transform identifications
    /// \param[in] node The node being visited
    /// \param[in,out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<MindDataNode> node, bool *const modified) override;
#endif

    /// \brief Getter
    std::vector<std::pair<std::shared_ptr<MappableSourceNode>, std::shared_ptr<DatasetNode>>> cache_pairs() {
      return cache_pairs_;
    }

    /// \brief Getter
    std::vector<std::shared_ptr<DatasetNode>> cached_nodes() { return cached_nodes_; }

    /// \brief Getter
    std::shared_ptr<SamplerObj> sampler() { return sampler_; }

   private:
    bool is_caching_;
    std::shared_ptr<MappableSourceNode> leaf_node_;
    std::shared_ptr<SamplerObj> sampler_;
    // The two nodes that work together to establish the cache transform
    std::vector<std::shared_ptr<DatasetNode>> cached_nodes_;
    std::vector<std::pair<std::shared_ptr<MappableSourceNode>, std::shared_ptr<DatasetNode>>> cache_pairs_;
  };

 public:
  /// \brief Constructor
  CacheTransformPass();

  /// \brief Destructor
  ~CacheTransformPass() = default;

  /// \brief Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
  /// \param[in,out] tree The tree to operate on.
  /// \param[in,out] Indicate of the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;

 private:
  /// \brief Helper function to execute mappable cache transformation.
  ///
  ///     Input:
  ///       Sampler
  ///         |
  ///       LeafNode --> OtherNodes --> CachedNode (cache_ = DatasetCache)
  ///
  ///     Transformed:
  ///       Sampler --> CacheLookupNode ------------------------->
  ///                           |                                |
  ///                           |                           CacheMergeNode
  ///                           |                                |
  ///                           LeafNode --> OtherNodes --> CachedNode
  ///
  /// \param[in] leaf_node The leaf node in the transform
  /// \param[in] cached_node The node with cache attribute which is involved in the cache transform
  /// \return Status The status code returned
  Status InjectMappableCacheNode(std::shared_ptr<MappableSourceNode> leaf_node,
                                 std::shared_ptr<DatasetNode> cached_node);

  /// \brief Helper function to execute non-mappable cache transformation.
  ///
  ///     Input:
  ///       LeafNode --> OtherNodes --> CachedNode (cache_ = DatasetCache)
  ///
  ///     Transformed:
  ///                                                   Sampler
  ///                                                      |
  ///       LeafNode --> OtherNodes --> CachedNode --> CacheNode
  ///
  /// \param[in] cached_node The node with cache attribute which is involved in the cache transform
  /// \param[in] sampler The sampler saved for non-mappable leaf nodes during the CachePass
  /// \return Status The status code returned
  Status InjectNonMappableCacheNode(std::shared_ptr<DatasetNode> cached_node, std::shared_ptr<SamplerObj> sampler);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_
