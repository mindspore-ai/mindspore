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
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"

#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#endif

namespace mindspore {
namespace dataset {

// Constructor
CacheTransformPass::CachePass::CachePass() : is_caching_(false), leaf_op_(nullptr) {}

// Identifies the subtree below this node as a cached descendant tree.
Status CacheTransformPass::CachePass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  *modified = false;
  MS_LOG(INFO) << "Cache transform pass: CacheOp found, identified descendant tree.";
  if (is_caching_) {
    RETURN_STATUS_UNEXPECTED("Nested cache operations is not supported!");
  }
  is_caching_ = true;
  return Status::OK();
}

// Resets the tracking of the cache within the tree and assigns the operators that will be involved in a cache
// transformation
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  *modified = false;
  is_caching_ = false;  // We a no longer in a cache subtree.  clear the flag.
  if (leaf_op_) {
    MS_LOG(INFO) << "Cache transform pass: Set up transformation nodes for mappable cache.";
    // Assign the leaf op into the transform pass, using move to null our copy of it, and also assign the cache op,
    // using base class pointers.
    AddMappableCacheOperators(std::move(leaf_op_), node);
  } else {
    // If there was no leaf_op set, then this is a non-mappable scenario.

    if (sampler_) {
      // Grab the sampler that was saved from the leaf and plug it into the cache op
      node->SetSampler(std::move(sampler_));
      MS_LOG(INFO) << "Cache transform pass: Set up cache sampler from non-mappable leaf.";
    } else {
      // We're a cache op but no sampler was saved from leaf, so create a default sampler
      int64_t num_samples = 0;
      int64_t start_index = 0;
      sampler_ = std::make_shared<SequentialSampler>(num_samples, start_index);
      node->SetSampler(std::move(sampler_));
      MS_LOG(INFO) << "Cache transform pass: Creating default sequential sampler for cache op.";
    }

    // Get the computed check sum from all ops in our cache path below us and ask the cache op to create it's cache
    uint32_t cache_crc = DatasetOp::GenerateCRC(node);
    RETURN_IF_NOT_OK(node->CreateCache(cache_crc));
  }

  return Status::OK();
}

// Common code for mappable leaf setup.
Status CacheTransformPass::CachePass::MappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op) {
  // If a leaf has already been assigned, then we have more than one leaf inside this cache descendant tree.
  if (is_caching_ && leaf_op_) {
    RETURN_STATUS_UNEXPECTED("There is currently no support for multiple leaf nodes under cache.");
  }

  // If we are a leaf in the caching path, then save this leaf.
  if (is_caching_) {
    MS_LOG(DEBUG) << "Cache transform pass: Mappable leaf in a cache descendant tree detected";
    leaf_op_ = std::move(leaf_op);
  }
  return Status::OK();
}

// Common code for non mappable leaf setup.
Status CacheTransformPass::CachePass::NonMappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op) {
  // If a leaf has already been assigned, then we have more than one leaf inside this cache descendant tree.
  if (is_caching_ && leaf_op_) {
    RETURN_STATUS_UNEXPECTED("There is currently no support for multiple leaf nodes under cache.");
  }

  // Sampler for non mapable dataset only works if there is a downstream cache. Remove it from the leaf
  // as save it for use by cache op in ascendant tree.
  if (is_caching_) {
    RETURN_IF_NOT_OK(leaf_op->FetchRemoveSampler(&sampler_));
    MS_LOG(DEBUG) << "Cache transform pass: Non mappable leaf in a cache descendant tree detected";
  } else {
    // If we are a non-mappable leaf and are not in a cache tree, then this sampler is not used so we can
    // remove it here.  The leaf itself will provide it's own methods of fetching the data (not sampler-based)
    std::shared_ptr<Sampler> sampler_from_leaf;
    RETURN_IF_NOT_OK(leaf_op->FetchRemoveSampler(&sampler_from_leaf));
  }
  return Status::OK();
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified) {
  if (is_caching_) {
    // If we are a TF Reader in a caching tree, then change our config so that it becomes a basic
    // TF reader that parses all files.  Selection of data will come from the sampler on the cache instead.
    node->MakeSimpleProducer();
  }
  return NonMappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<RandomDataOp> node, bool *modified) {
  return NonMappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<MnistOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<CocoOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

#ifdef ENABLE_PYTHON
// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache transform identification
Status CacheTransformPass::CachePass::RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}
#endif

// Assigns the leaf and cache operators that are involved in a cache transformation
void CacheTransformPass::CachePass::AddMappableCacheOperators(std::shared_ptr<DatasetOp> leaf_op,
                                                              std::shared_ptr<CacheOp> cache_op) {
  cache_pairs_.push_back(std::make_pair(leaf_op, cache_op));
}

// constructor
CacheTransformPass::CacheTransformPass() {}

// Runs a cache_pass first to set up the transformation nodes, and then drives any of these transformations
Status CacheTransformPass::RunOnTree(ExecutionTree *tree, bool *modified) {
  MS_LOG(INFO) << "Pre pass: Cache transform pass started.";
  // Create the cache pass and run it.  The cache pass identifies and creates the leaf/cache pairs that we will
  // use to execute a transform.
  CachePass cache_pass = CachePass();
  RETURN_IF_NOT_OK(cache_pass.Run(tree, modified));

  // Then, execute the transform for each pair
  for (auto cache_pair : cache_pass.cache_pairs()) {
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
}  // namespace dataset
}  // namespace mindspore
