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

#include "minddata/dataset/engine/ir/cache/pre_built_dataset_cache.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"

namespace mindspore {
namespace dataset {
/// Method to initialize the DatasetCache by creating an instance of a CacheClient
/// \return Status Error code
Status PreBuiltDatasetCache::Build() {
  // we actually want to keep a reference of the runtime object so it can be shared by different pipelines
  return Status::OK();
}

Status PreBuiltDatasetCache::CreateCacheOp(int32_t num_workers, std::shared_ptr<DatasetOp> *const ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "Cache client has not been created yet.");
  std::shared_ptr<CacheOp> cache_op = nullptr;
  RETURN_IF_NOT_OK(CacheOp::Builder().SetNumWorkers(num_workers).SetClient(cache_client_).Build(&cache_op));
  *ds = cache_op;

  return Status::OK();
}

Status PreBuiltDatasetCache::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["session_id"] = cache_client_->session_id();
  args["cache_memory_size"] = cache_client_->GetCacheMemSz();
  args["spill"] = cache_client_->isSpill();
  args["num_connections"] = cache_client_->GetNumConnections();
  args["prefetch_size"] = cache_client_->GetPrefetchSize();
  *out_json = args;
  return Status::OK();
}

Status PreBuiltDatasetCache::CreateCacheLookupOp(int32_t num_workers, std::shared_ptr<DatasetOp> *ds,
                                                 std::shared_ptr<SamplerObj> sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "Cache client has not been created yet.");
  std::shared_ptr<CacheLookupOp> lookup_op = nullptr;
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler->SamplerBuild(&sampler_rt));

  RETURN_IF_NOT_OK(CacheLookupOp::Builder()
                     .SetNumWorkers(num_workers)
                     .SetClient(cache_client_)
                     .SetSampler(sampler_rt)
                     .Build(&lookup_op));
  *ds = lookup_op;

  return Status::OK();
}

Status PreBuiltDatasetCache::CreateCacheMergeOp(int32_t num_workers, std::shared_ptr<DatasetOp> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "Cache client has not been created yet.");
  std::shared_ptr<CacheMergeOp> merge_op = nullptr;
  RETURN_IF_NOT_OK(CacheMergeOp::Builder().SetNumWorkers(num_workers).SetClient(cache_client_).Build(&merge_op));
  *ds = merge_op;

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
