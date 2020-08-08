/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/util/cache_pool.h"
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
class BaseRequest;
/// \brief A server which provides CacheService services.
class CacheServer : public Service {
 public:
  friend class Services;
  using cache_index = std::map<connection_id_type, std::unique_ptr<CacheService>>;

  CacheServer(const CacheServer &) = delete;
  CacheServer &operator=(const CacheServer &) = delete;
  CacheServer(CacheServer &&) = delete;
  CacheServer &operator=(CacheServer &) = delete;
  static CacheServer &GetInstance() noexcept { return Services::getCacheServer(); }
  Status DoServiceStart() override;
  Status DoServiceStop() override;
  ~CacheServer() { (void)ServiceStop(); }

  /// \brief For the current demonstration, a cache client contacts cache server using a Queue.
  /// \param rq
  /// \return Status object
  Status PushRequest(BaseRequest *rq) {
    RETURN_UNEXPECTED_IF_NULL(rq);
    RETURN_IF_NOT_OK(cache_q_->Add(rq));
    return Status::OK();
  }

 private:
  mutable RWLock rwLock_;
  std::string top_;
  cache_index all_caches_;
  std::shared_ptr<Queue<BaseRequest *>> cache_q_;
  TaskGroup vg_;
  int32_t num_workers_;

  /// \brief Constructor
  /// \param spill_path Top directory for spilling buffers to.
  /// \param num_workers Number of threads for handling requests.
  explicit CacheServer(const std::string &spill_path, int32_t num_workers = 3);

  /// \brief Locate a cache service from connection id.
  /// \return Pointer to cache service. Null if not found
  CacheService *GetService(connection_id_type id) const;

  /// \brief Create a cache service. We allow multiple clients to create the same cache service.
  /// Subsequent duplicate requests are ignored. The first cache client to create the service will be given
  /// a special unique cookie.
  /// \param[in] connection_id This is from a Cache client.
  /// \param[in] cache_mem_sz
  /// \param[in] flag
  /// \param[out] out_cookie Only the first cache client will be given a special cookie to identify the creator
  /// \return Status object
  Status CreateService(connection_id_type connection_id, uint64_t cache_mem_sz, BaseRequest::CreateCacheFlag flag,
                       std::string *out_cookie);

  /// \brief Entry point for all server threads.
  Status ServerRequest();
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CACHE_TENSOR_H_
