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

#include <memory>
#include "minddata/dataset/engine/opt/pre/cache_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"

namespace mindspore {
namespace dataset {

// Constructor
CachePass::CachePass(CacheTransformPass *transform_pass)
    : transform_pass_(transform_pass), is_caching_(false), leaf_op_(nullptr) {}

// Identifies the subtree below this node as a cached descendant tree.
Status CachePass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
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
Status CachePass::RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  *modified = false;
  is_caching_ = false;  // We a no longer in a cache subtree.  clear the flag.
  if (leaf_op_) {
    MS_LOG(INFO) << "Cache transform pass: Set up transformation nodes for mappable cache.";
    // Assign the leaf op into the transform pass, using move to null our copy of it, and also assign the cache op,
    // using base class pointers.
    transform_pass_->AddMappableCacheOperators(std::move(leaf_op_), node);
  } else {
    // If there was no leaf_op set, then this is a non-mappable scenario.

    if (sampler_) {
      // Grab the sampler that was saved from the leaf and plug it into the cache op
      node->SetSampler(std::move(sampler_));
      MS_LOG(INFO) << "Cache transform pass: Set up cache sampler from non-mappable leaf.";
    } else {
      // We're a cache op but no sampler was saved from leaf, so create a default sampler
      const int64_t num_samples = 0;
      const int64_t start_index = 0;
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
Status CachePass::MappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op) {
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
Status CachePass::NonMappableCacheLeafSetup(std::shared_ptr<DatasetOp> leaf_op) {
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

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified) {
  if (is_caching_) {
    // If we are a TF Reader in a caching tree, then change our config so that it becomes a basic
    // TF reader that parses all files.  Selection of data will come from the sampler on the cache instead.
    node->MakeSimpleProducer();
  }
  return NonMappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<RandomDataOp> node, bool *modified) {
  return NonMappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<MnistOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<CocoOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}

// Perform leaf node cache tranform identifications
Status CachePass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) {
  return MappableCacheLeafSetup(std::static_pointer_cast<DatasetOp>(node));
}
}  // namespace dataset
}  // namespace mindspore
