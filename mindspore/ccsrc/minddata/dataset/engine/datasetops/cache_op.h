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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_OP_H_

#include <atomic>
#include <string>
#include <utility>
#include <memory>
#include "minddata/dataset/engine/datasetops/cache_base_op.h"

namespace mindspore {
namespace dataset {
/// \brief CacheOp provides a memory/disk cache that acts as a save-point within a non-mappable dataset.
/// \note For mappable dataset, please see CacheLookupOp.
/// \see CacheLookupOp
class CacheOp : public CacheBase, public RandomAccessOp {
 public:
  // This CacheOp is for non-mappable case where it is divided into two phases.
  // The first phase is we cache all the rows from the child (and let the cache server
  // assigns row id). No read access in the first phase. Once the cache is fully built,
  // we switch to second phase and fetch requests from the sampler.
  enum class Phase : uint8_t { kBuildPhase = 0, kFetchPhase = 1 };
  constexpr static int32_t kPhaseCheckIntervalInMilliSec = 100;

  /// \brief Constructor of CacheOp
  /// \note The builder class should be used to call it.
  /// \param num_workers The number of worker threads.
  /// \param op_connector_size The size of each queue in the connector.
  CacheOp(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<CacheClient> cache_client,
          std::shared_ptr<SamplerRT> sampler);

  // Destructor
  ~CacheOp();

  /// \brief Base-class override for special eoe handler.
  /// \notes CacheOp must override this because it shall not perform default handling of eoe. Instead
  ///     the CacheOp manages actions related to the end of the epoch.
  /// \return Status The status code returned
  Status EoeReceived(int32_t worker_id) override;

  /// \brief Base-class override for handling cases when an eof is received.
  /// \param worker_id - The worker id
  /// \return Status The status code returned
  Status EofReceived(int32_t worker_id) override;

  // \brief Class functor operator ().
  /// \return Status The status code returned
  Status operator()() override;

  /// \brief Entry function for worker thread that fetch rows from CacheLookupOp
  /// \param workerId
  /// \return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  /// \brief Base-class override for handling cases if we allow cache miss.
  bool AllowCacheMiss() override { return false; }

  /// \brief Base-class override for the name of this operator.
  std::string Name() const override { return kCacheOp; }

  /// \brief Perform specific post-operations on CacheOp
  /// \return Status The status code returned
  Status PrepareOperator() override;

 private:
  WaitPost rows_cache_done_;
  std::atomic<int64_t> num_guys_in_;
  Phase phase_;

  QueueList<TensorRow> cache_workers_in_queue_;
  /// \brief The main thread will wait until all the rows are cached and will start the handshake with the sampler.
  /// \return Status object
  Status WaitForCachingAllRows();

  Status CacheAllRowsMaster();

  /// \brief For non-mappable dataset, there is a build phase where we cache all the rows.
  /// \return Status object
  Status CacheAllRows(int32_t worker_id);
  Status RegisterResources() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_OP_H_
