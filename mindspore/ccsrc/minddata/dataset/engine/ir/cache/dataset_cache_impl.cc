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

#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"

namespace mindspore {
namespace dataset {
/// Method to initialize the DatasetCache by creating an instance of a CacheClient
/// \return Status Error code
Status DatasetCacheImpl::Build() {
  // The same DatasetCache instance can be re-used for multiple pipelines for cache sharing,
  // in this case, cache_client_ object might have been created.
  if (cache_client_) return Status::OK();

  CacheClient::Builder builder;
  builder.SetSessionId(session_id_).SetCacheMemSz(cache_mem_sz_).SetSpill(spill_);
  if (hostname_) {
    (void)builder.SetHostname(hostname_.value());
  }
  if (port_) {
    (void)builder.SetPort(port_.value());
  }
  if (num_connections_) {
    (void)builder.SetNumConnections(num_connections_.value());
  }
  if (prefetch_sz_) {
    (void)builder.SetPrefetchSize(prefetch_sz_.value());
  }
  return builder.Build(&cache_client_);
}

Status DatasetCacheImpl::CreateCacheOp(int32_t num_workers, int32_t connector_queue_size,
                                       std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetOp> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "CacheOp requires a CacheClient, but got nullptr.");
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler->SamplerBuild(&sampler_rt));
  std::shared_ptr<CacheOp> cache_op =
    std::make_shared<CacheOp>(num_workers, connector_queue_size, cache_client_, std::move(sampler_rt));
  *ds = cache_op;

  return Status::OK();
}

Status DatasetCacheImpl::CreateCacheLookupOp(int32_t num_workers, int32_t connector_queue_size,
                                             std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetOp> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "CacheLookupOp requires a CacheClient, but got nullptr.");
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler->SamplerBuild(&sampler_rt));
  std::shared_ptr<CacheLookupOp> lookup_op =
    std::make_shared<CacheLookupOp>(num_workers, connector_queue_size, cache_client_, std::move(sampler_rt));
  *ds = lookup_op;

  return Status::OK();
}

Status DatasetCacheImpl::CreateCacheMergeOp(int32_t num_workers, int32_t connector_queue_size,
                                            std::shared_ptr<DatasetOp> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "CacheMergeOp requires a CacheClient, but got nullptr.");
  std::shared_ptr<CacheMergeOp> merge_op =
    std::make_shared<CacheMergeOp>(num_workers, connector_queue_size, num_workers, cache_client_);
  *ds = merge_op;

  return Status::OK();
}

Status DatasetCacheImpl::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["session_id"] = session_id_;
  args["cache_memory_size"] = cache_mem_sz_;
  args["spill"] = spill_;
  if (hostname_) args["hostname"] = hostname_.value();
  if (port_) args["port"] = port_.value();
  if (num_connections_) args["num_connections"] = num_connections_.value();
  if (prefetch_sz_) args["prefetch_size"] = prefetch_sz_.value();
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
