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

#include <vector>
#include "minddata/dataset/engine/opt/pre/cache_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"

namespace mindspore {
namespace dataset {

// constructor
CacheTransformPass::CacheTransformPass() {}

// Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
Status CacheTransformPass::RunOnTree(ExecutionTree *tree, bool *modified) {
  MS_LOG(INFO) << "Pre pass: Cache transform pass started.";
  // Create the cache pass and run it.  The cache pass identifies and creates the leaf/cache pairs that we will
  // use to execute a transform.
  std::unique_ptr<Pass> cache_pass = std::make_unique<CachePass>(this);
  RETURN_IF_NOT_OK(cache_pass->Run(tree, modified));

  // Then, execute the transform for each pair
  for (auto cache_pair : cache_pairs_) {
    MS_LOG(DEBUG) << "Cache transform pass: Executing a cache op mappable transform.";
    ExecuteCacheTransform(tree, cache_pair.first, cache_pair.second, cache_pair.second->cache_client());
  }
  MS_LOG(INFO) << "Pre pass: Cache transform pass complete.";
  return Status::OK();
}

// Helper function to execute the cache transformation.
Status CacheTransformPass::ExecuteCacheTransform(ExecutionTree *tree, std::shared_ptr<DatasetOp> leaf_op,
                                                 std::shared_ptr<DatasetOp> cache_op,
                                                 std::shared_ptr<CacheClient> cache_client) {
  // Get local pointers the child/parent of the cache op.  It's possible that the parent is null if the cache was
  // the root node.  It is also possible that cache_child == leaf_op
  std::shared_ptr<DatasetOp> cache_child = cache_op->child(0);
  DatasetOp *cache_parent = nullptr;
  cache_op->Parent(&cache_parent, 0);  // fetch the cache op's parent

  // Extract the sampler from the leaf.  We will overwrite this sampler with the lookup op later.
  std::shared_ptr<Sampler> leaf_sampler = leaf_op->sampler();

  // Construct the merge op with defaults
  std::shared_ptr<CacheMergeOp> merge_op;
  CacheMergeOp::Builder merge_builder;
  RETURN_IF_NOT_OK(merge_builder.SetClient(cache_client).Build(&merge_op));
  RETURN_IF_NOT_OK(tree->AssociateNode(merge_op));

  // Construct the cache lookup op with defaults
  std::shared_ptr<CacheLookupOp> cache_lookup_op;
  CacheLookupOp::Builder lookup_builder;
  RETURN_IF_NOT_OK(lookup_builder.SetClient(cache_client).SetSampler(std::move(leaf_sampler)).Build(&cache_lookup_op));
  RETURN_IF_NOT_OK(tree->AssociateNode(cache_lookup_op));

  // Overwrite the old sampler in this leaf op to become the lookup op
  leaf_op->SetSampler(cache_lookup_op);

  // If the cache had a parent, then go into that parent to remove the cache from it's child list and then
  // replace it with the merge op.
  if (cache_parent != nullptr) {
    RETURN_IF_NOT_OK(cache_parent->RemoveChild(cache_op));
    RETURN_IF_NOT_OK(cache_parent->AddChild(merge_op));
  } else {
    // If we didn't have a parent, then the merge op is the root node
    RETURN_IF_NOT_OK(tree->AssignRoot(merge_op));
  }

  // Set the cache op to no longer be a parent over it's child. This will fully disconnect the old cache op.
  // We maintain a local pointer to the old child though.
  RETURN_IF_NOT_OK(cache_op->RemoveChild(cache_child));

  // Connect the merge op
  RETURN_IF_NOT_OK(merge_op->AddChild(std::move(cache_lookup_op)));
  RETURN_IF_NOT_OK(merge_op->AddChild(std::move(cache_child)));

  // At this point, the cache op has already had it's children and parents taken away. Calling remove
  // on it at this point will not do any node hookups, and instead set internal fields to invalid.
  RETURN_IF_NOT_OK(cache_op->Remove());

  return Status::OK();
}

// Assigns the leaf and cache operators that are involved in a cache transformation
void CacheTransformPass::AddMappableCacheOperators(std::shared_ptr<DatasetOp> leaf_op,
                                                   std::shared_ptr<CacheOp> cache_op) {
  cache_pairs_.push_back(std::make_pair(leaf_op, cache_op));
}
}  // namespace dataset
}  // namespace mindspore
