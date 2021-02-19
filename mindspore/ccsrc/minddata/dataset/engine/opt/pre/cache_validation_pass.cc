/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/opt/pre/cache_validation_pass.h"

#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/filter_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {

// Constructor
CacheValidationPass::CacheValidationPass() : is_cached_(false), is_mappable_(false) {}

// Returns an error if BatchNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<BatchNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<BatchNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("BatchNode is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("BatchNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if ConcatNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<ConcatNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<ConcatNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ConcatNode is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("ConcatNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if FilterNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<FilterNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<FilterNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("FilterNode is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("FilterNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if SkipNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<SkipNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<SkipNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("SkipNode is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("SkipNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if TakeNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<TakeNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<TakeNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("TakeNode (possibly from Split) is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("TakeNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if ZipNode exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<ZipNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<ZipNode>): visiting " << node->Name() << ".";
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ZipNode is not supported as a descendant operator under a cache.");
  }
  if (node->IsCached()) {
    RETURN_STATUS_UNEXPECTED("ZipNode cannot be cached.");
  }
  return Status::OK();
}

// Returns an error if MapNode with non-deterministic tensor operations exists under a cache
Status CacheValidationPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<MapNode>): visiting " << node->Name() << ".";
  if (node->IsCached()) {
    if (is_cached_) {
      RETURN_STATUS_UNEXPECTED("Nested cache operations over MapNode is not supported.");
    }
    // If Map is created to be cached, set the flag indicating we found an operation with a cache.
    is_cached_ = true;

    // This is temporary code.
    // Because the randomness of its tensor operations is not known in TensorOperation form until we convert them
    // to TensorOp, we need to check the randomness in MapNode::Build().
    // By setting this MapNode is under a cache, we will check the randomness of its tensor operations without the need
    // to walk the IR tree again.
    node->Cached();

    auto tfuncs = node->TensorOperations();
    for (size_t i = 0; i < tfuncs.size(); i++) {
      if (tfuncs[i]->IsRandomOp()) {
        RETURN_STATUS_UNEXPECTED("MapNode containing random operation is not supported as a descendant of cache.");
      }
    }
  }
  return Status::OK();
}

// Flag an error if we have a cache over another cache
Status CacheValidationPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::Visit(<DatasetNode>): visiting " << node->Name() << ".";
  if (node->IsCached()) {
    if (is_cached_) {
      RETURN_STATUS_UNEXPECTED("Nested cache operations over " + node->Name() + " is not supported.");
    }
    // If this node is created to be cached, set the flag.
    is_cached_ = true;
  }
  if (node->IsLeaf() && node->IsMappableDataSource()) {
    is_mappable_ = true;
  }
  return Status::OK();
}

// Returns an error if MappableSource <- Repeat <- Node with a cache
// Because there is no operator in the cache hit stream to consume EoEs, caching above repeat causes problem.
Status CacheValidationPass::VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::VisitAfter(<RepeatNode>): visiting " << node->Name() << ".";
  if (is_cached_ && is_mappable_) {
    RETURN_STATUS_UNEXPECTED("A cache over a RepeatNode of a mappable dataset is not supported.");
  }
  return Status::OK();
}

Status CacheValidationPass::VisitAfter(std::shared_ptr<TFRecordNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::VisitAfter(<TFRecordNode>): visiting " << node->Name() << ".";
  if (!is_cached_) {
    // If we are not in a cache path, then we must validate the file-based sharding config.
    // If we are in a cache path, there is no file-based sharding so the check is not required.
    if (!node->shard_equal_rows_ && node->dataset_files_.size() < static_cast<uint32_t>(node->num_shards_)) {
      RETURN_STATUS_UNEXPECTED("Invalid file, not enough tfrecord files provided.\n");
    }
  }
  // Reset the flag when this node is cached and is already visited
  if (node->IsCached()) {
    is_cached_ = false;
  }
  return Status::OK();
}

Status CacheValidationPass::VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) {
  MS_LOG(DEBUG) << "CacheValidationPass::VisitAfter(<DatasetNode>): visiting " << node->Name() << ".";
  // Reset the flag when all descendants are visited
  if (node->IsCached()) {
    is_cached_ = false;
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
