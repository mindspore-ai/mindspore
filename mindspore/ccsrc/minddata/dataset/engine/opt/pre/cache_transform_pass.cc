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
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/ir/datasetops/cache_lookup_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_merge_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_node.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#endif
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"

namespace mindspore {
namespace dataset {

// Constructor
CacheTransformPass::CachePass::CachePass() : is_caching_(false), leaf_node_(nullptr), sampler_(nullptr) {}

// Identifies the subtree below this node as a cached descendant tree.
// Note that this function will only get called on non-leaf nodes.
// For leaf nodes, the other Visit with NonMappableSourceNode or MappableSourceNode argument will be called instead.
Status CacheTransformPass::CachePass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = false;
  if (node->IsCached()) {
    MS_LOG(INFO) << "Cache transform pass: CacheOp found, identified descendant tree.";
    is_caching_ = true;
  }
  return Status::OK();
}

// Resets the tracking of the cache within the tree and assigns the nodes that will be involved in a cache
// transformation
Status CacheTransformPass::CachePass::VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = false;
  if (node->IsCached()) {
    is_caching_ = false;  // We a no longer in a cache subtree.  clear the flag.
    if (leaf_node_) {
      MS_LOG(INFO) << "Cache transform pass: Set up transformation nodes for mappable cache.";
      // Assign the leaf node into the transform pass, using move to null our copy of it,
      // and also assign the cached node, using base class pointers.
      // In the cases where cache is directly injected after the leaf node, these two nodes might be the same.
      cache_pairs_.push_back(std::make_pair(std::move(leaf_node_), node));
    } else {
      // If there was no leaf_node_ set, then this is a non-mappable scenario.
      // We only assign the cached node in this case.
      cached_nodes_.push_back(node);
    }
  }

  return Status::OK();
}

#ifndef ENABLE_ANDROID
// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) {
  if (node->IsCached()) {
    MS_LOG(INFO) << "Cache transform pass: CacheOp found, identified descendant tree.";
    is_caching_ = true;
  }
  // Cache might also be injected to the non-leaf node upper in the tree, so is_caching_ might also be set to true
  // by the other Visit() with DatasetNode argument
  if (is_caching_) {
    MS_LOG(DEBUG) << "Cache transform pass: Non mappable leaf in a cache descendant tree detected";
    // If a leaf has already been assigned, then we have more than one leaf inside this cache descendant tree.
    if (leaf_node_) {
      return Status(StatusCode::kMDNotImplementedYet, __LINE__, __FILE__,
                    "There is currently no support for multiple leaf nodes under cache.");
    }
    // Set up a sampler here to be used by cache if we are a non-mappable leaf in a caching tree.
    // Node that sampler for non mappable dataset only works if there is a downstream cache.
    RETURN_IF_NOT_OK(node->SetupSamplerForCache(&sampler_));
    // If we are a non-mappable source node in a caching tree, then change our config so that it becomes a basic
    // source node that parses all files. Selection of data will come from the sampler on the cache instead.
    RETURN_IF_NOT_OK(node->MakeSimpleProducer());
  }
  return Status::OK();
}
#endif

Status CacheTransformPass::CachePass::Visit(std::shared_ptr<RandomNode> node, bool *const modified) {
  if (node->IsCached()) {
    MS_LOG(INFO) << "Cache transform pass: CacheOp found, identified descendant tree.";
    is_caching_ = true;
  }
  // Cache might also be injected to the non-leaf node upper in the tree, so is_caching_ might also be set to true
  // by the other Visit() with DatasetNode argument
  if (is_caching_) {
    MS_LOG(DEBUG) << "Cache transform pass: Non mappable leaf in a cache descendant tree detected";
    // If a leaf has already been assigned, then we have more than one leaf inside this cache descendant tree.
    if (leaf_node_) {
      return Status(StatusCode::kMDNotImplementedYet, __LINE__, __FILE__,
                    "There is currently no support for multiple leaf nodes under cache.");
    }
    // Set up a sampler here to be used by cache if we are a non-mappable leaf in a caching tree.
    // Node that sampler for non mappable dataset only works if there is a downstream cache.
    RETURN_IF_NOT_OK(node->SetupSamplerForCache(&sampler_));
  }
  return Status::OK();
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) {
  if (node->IsCached()) {
    MS_LOG(INFO) << "Cache transform pass: CacheOp found, identified descendant tree.";
    is_caching_ = true;
  }
  // Cache might also be injected to the non-leaf node upper in the tree, so is_caching_ might also be set to true
  // by the other Visit() with DatasetNode argument
  if (is_caching_) {
    MS_LOG(DEBUG) << "Cache transform pass: Mappable leaf in a cache descendant tree detected";
    // If a leaf has already been assigned, then we have more than one leaf inside this cache descendant tree.
    if (leaf_node_) {
      return Status(StatusCode::kMDNotImplementedYet, __LINE__, __FILE__,
                    "There is currently no support for multiple leaf nodes under cache.");
    }
    // If we are a leaf in the caching path, then save this leaf
    leaf_node_ = node;
  }
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::Visit(std::shared_ptr<MindDataNode> node, bool *const modified) {
  if (node->IsCached() || is_caching_) {
    return Status(StatusCode::kMDNotImplementedYet, __LINE__, __FILE__,
                  "There is currently no support for MindRecordOp under cache.");
  }
  return Status::OK();
}
#endif

#ifdef ENABLE_PYTHON
// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) {
  if (node->IsCached() || is_caching_) {
    return Status(StatusCode::kMDNotImplementedYet, __LINE__, __FILE__,
                  "There is currently no support for GeneratorOp under cache.");
  }
  return Status::OK();
}
#endif

// constructor
CacheTransformPass::CacheTransformPass() {}

// Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
Status CacheTransformPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: Cache transform pass started.";
  // Create the cache pass and run it.  The cache pass identifies and creates the leaf/cache pairs that we will
  // use to execute a transform.
  CachePass cache_pass = CachePass();
  RETURN_IF_NOT_OK(cache_pass.Run(root_ir, modified));

  // Execute the transform for non-mappable cache
  for (auto cached_node : cache_pass.cached_nodes()) {
    MS_LOG(DEBUG) << "Cache transform pass: Injecting a non-mappable cache node.";
    RETURN_IF_NOT_OK(InjectNonMappableCacheNode(cached_node, cache_pass.sampler()));
  }

  // Execute the transform for mappable cache
  for (auto cache_pair : cache_pass.cache_pairs()) {
    MS_LOG(DEBUG) << "Cache transform pass: Injecting a mappable cache node.";
    RETURN_IF_NOT_OK(InjectMappableCacheNode(cache_pair.first, cache_pair.second));
  }
  MS_LOG(INFO) << "Pre pass: Cache transform pass complete.";
  return Status::OK();
}

// Helper function to execute mappable cache transformation.
// Input:
//   Sampler
//     |
//   LeafNode --> OtherNodes --> CachedNode (cache_ = DatasetCache)
//
// Transformed:
//   Sampler --> CacheLookupNode ------------------------->
//                       |                                |
//                       |                           CacheMergeNode
//                       |                                |
//                       LeafNode --> OtherNodes --> CachedNode
Status CacheTransformPass::InjectMappableCacheNode(std::shared_ptr<MappableSourceNode> leaf_node,
                                                   std::shared_ptr<DatasetNode> cached_node) {
  // Create a cache merge node with defaults
  auto cache_merge_node = std::make_shared<CacheMergeNode>(nullptr, cached_node->GetDatasetCache());
  // Insert the cache merge node to become the cached_node's parent
  RETURN_IF_NOT_OK(cached_node->InsertAbove(cache_merge_node));

  // Extract the sampler from the leaf.  We will overwrite this sampler with the lookup op later.
  std::shared_ptr<SamplerObj> leaf_sampler = leaf_node->Sampler();
  // Create a cache lookup node with leaf_node's sampler
  auto cache_lookup_node = std::make_shared<CacheLookupNode>(nullptr, leaf_sampler, cached_node->GetDatasetCache());
  // Insert the cache lookup node as the first child of cache merge node
  RETURN_IF_NOT_OK(cache_merge_node->InsertChildAt(0, cache_lookup_node));
  // Overwrite the old sampler in this leaf node to become the cache lookup node
  leaf_node->SetSampler(std::static_pointer_cast<SamplerObj>(cache_lookup_node));
  return Status::OK();
}

// Helper function to execute non-mappable cache transformation.
// Input:
//   LeafNode --> OtherNodes --> CachedNode (cache_ = DatasetCache)
//
// Transformed:
//                                               Sampler
//                                                  |
//   LeafNode --> OtherNodes --> CachedNode --> CacheNode
Status CacheTransformPass::InjectNonMappableCacheNode(std::shared_ptr<DatasetNode> cached_node,
                                                      std::shared_ptr<SamplerObj> sampler) {
  // Create a cache node using the sampler we saved from the leaf
  auto cache_node = std::make_shared<CacheNode>(nullptr, sampler, cached_node->GetDatasetCache());
  // Insert the cache node to become the cached_node's parent
  RETURN_IF_NOT_OK(cached_node->InsertAbove(cache_node));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
