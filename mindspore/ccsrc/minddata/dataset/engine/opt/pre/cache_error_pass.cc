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
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/opt/pre/cache_error_pass.h"

namespace mindspore {
namespace dataset {

// Constructor
CacheErrorPass::CacheErrorPass() : is_cached_(false) {}

// Identifies the subtree below this node as being cached
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  // Turn on the flag that we're under a merge op
  is_cached_ = true;
  return Status::OK();
}

// Returns an error if ZipOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<ZipOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ZipOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}

// Returns an error if MapOp with non-deterministic TensorOps exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<MapOp> node, bool *modified) {
  if (is_cached_) {
    auto tfuncs = node->TFuncs();
    for (size_t i = 0; i < tfuncs.size(); i++) {
      if (!tfuncs[i]->Deterministic()) {
        RETURN_STATUS_UNEXPECTED(
          "MapOp with non-deterministic TensorOps is currently not supported as a descendant of cache.");
      }
    }
  }
  return Status::OK();
}

// Returns an error if ConcatOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<ConcatOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ConcatOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}

#ifdef ENABLE_PYTHON
// Returns an error if FilterOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<FilterOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("FilterOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
