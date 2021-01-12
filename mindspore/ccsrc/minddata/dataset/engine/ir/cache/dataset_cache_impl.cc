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

#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
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
  if (hostname_) builder.SetHostname(hostname_.value());
  if (port_) builder.SetPort(port_.value());
  if (num_connections_) builder.SetNumConnections(num_connections_.value());
  if (prefetch_sz_) builder.SetPrefetchSize(prefetch_sz_.value());
  return builder.Build(&cache_client_);
}

Status DatasetCacheImpl::CreateCacheOp(int32_t num_workers, std::shared_ptr<DatasetOp> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_client_ != nullptr, "Cache client has not been created yet.");
  std::shared_ptr<CacheOp> cache_op = nullptr;
  RETURN_IF_NOT_OK(CacheOp::Builder().SetNumWorkers(num_workers).SetClient(cache_client_).Build(&cache_op));
  *ds = cache_op;

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
