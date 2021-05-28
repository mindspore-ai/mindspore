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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_PRE_BUILT_DATASET_CACHE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_PRE_BUILT_DATASET_CACHE_H_

#include <memory>
#include <string>
#include <utility>
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

namespace mindspore {
namespace dataset {
/// DatasetCache is the IR of CacheClient
class PreBuiltDatasetCache : public DatasetCacheImpl {
 public:
  /// \brief Constructor
  /// \param cc a pre-built cache client
  explicit PreBuiltDatasetCache(std::shared_ptr<CacheClient> cc)
      : DatasetCacheImpl(cc->session_id(), cc->GetCacheMemSz(), cc->isSpill(), StringToChar(cc->GetHostname()),
                         cc->GetPort(), cc->GetNumConnections(), cc->GetPrefetchSize()) {
    cache_client_ = std::move(cc);
  }

  ~PreBuiltDatasetCache() = default;

  /// Method to initialize the DatasetCache by creating an instance of a CacheClient
  /// \return Status Error code
  Status Build() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_PRE_BUILT_DATASET_CACHE_H_
