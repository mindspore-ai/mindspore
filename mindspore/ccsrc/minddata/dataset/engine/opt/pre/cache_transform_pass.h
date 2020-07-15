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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_
#define DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_

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
 public:
  /// \brief Constructor
  CacheTransformPass();

  /// \brief Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
  /// \param[inout] tree The tree to operate on.
  /// \param[inout] Indicate of the tree was modified.
  /// \return Status The error code return
  Status RunOnTree(ExecutionTree *tree, bool *modified) override;

  /// \brief Assigns the leaf and cache operators that are involved in a cache transformation
  /// \param[in] leaf_op The leaf operator involved in the cache transform
  /// \param[in] cache_op The cache operator involved in the cache transform
  void AddMappableCacheOperators(std::shared_ptr<DatasetOp> leaf_op, std::shared_ptr<CacheOp> cache_op);

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

  // The two operators that work together to establish the cache transform
  std::vector<std::pair<std::shared_ptr<DatasetOp>, std::shared_ptr<CacheOp>>> cache_pairs_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_CACHE_TRANSFORM_PASS_H_
