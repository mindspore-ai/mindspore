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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_BASE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_BASE_OP_H_

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/connector.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/queue_map.h"
#include "minddata/dataset/util/semaphore.h"
#include "minddata/dataset/util/wait_post.h"
namespace mindspore {
namespace dataset {
/// \brief This is the base class for CacheOp and CacheLookupOp which share many similarities.
/// \see CacheOp
/// \see CacheLookupOp
class CacheBase : public ParallelOp {
 public:
  /// \brief Base class constructor
  /// \param num_workers Number of parallel workers
  /// \param op_connector_size Connector size
  /// \param cache_client CacheClient for communication to the CacheServer
  /// \param sampler Sampler which is mandatory
  CacheBase(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<CacheClient> cache_client,
            std::shared_ptr<SamplerRT> sampler);
  /// \brief Destructor
  ~CacheBase();

  /// \brief Overrides base class reset method.  When an operator does a reset, it cleans up any state
  /// info from it's previous execution and then initializes itself so that it can be executed
  /// again.
  /// \return Status The status code returned
  Status Reset() override;

  /// \brief A print method typically used for debugging
  /// \param out The output stream to write output to
  /// \param show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Gives a name to the class, typically used for debugging
  std::string Name() const override { return kCacheBase; }

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out reference to the output stream being overloaded
  /// \param mo reference to the CacheOp to display
  /// \return the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const CacheBase &mo) {
    mo.Print(out, false);
    return out;
  }

  /// \brief Getter for the cache client
  /// \return shared ptr to the cache client
  std::shared_ptr<CacheClient> GetCacheClient() { return cache_client_; }
  /// \brief Setter for the cache client
  void SetCacheClient(std::shared_ptr<CacheClient> cache_client) { cache_client_ = std::move(cache_client); }
  /// \brief Derived class must implement this method if a cache miss is treated as error
  virtual bool AllowCacheMiss() = 0;

 protected:
  constexpr static int32_t eoe_row_id = -1;
  int64_t row_cnt_;
  std::atomic<int64_t> num_cache_miss_;
  std::shared_ptr<CacheClient> cache_client_;
  std::unique_ptr<Connector<std::vector<row_id_type>>> keys_miss_;

  /// \brief Common function to register resources for interrupt
  /// \note Derived should override this function for extra resources to be registered
  virtual Status RegisterResources();
  /// \brief This function is called by main thread to send samples to the worker thread.
  /// \note It is a non-virtual function
  /// \return Status object
  Status FetchSamplesToWorkers();
  /// \brief This function is called by each worker to fetch rows from the cache server for a given set of
  /// sample row id's
  /// \return Status object
  Status FetchFromCache(int32_t worker_id);
  /// \brief Get the column map from cache server
  Status UpdateColumnMapFromCache();

 private:
  constexpr static int32_t connector_capacity_ = 1024;
  int32_t prefetch_size_;
  int32_t num_prefetchers_;
  QueueList<std::unique_ptr<IOBlock>> prefetch_queues_;
  QueueMap<row_id_type, TensorRow> prefetch_;

  /// \brief Prefetcher. It prefetch the rows from cache server
  /// \return Status object.
  Status Prefetcher(int32_t worker_id);
  /// \brief Functions used by prefetcher and WorkerEntry
  Status PrefetchRows(const std::vector<row_id_type> &keys, std::vector<row_id_type> *cache_miss);
  Status GetPrefetchRow(row_id_type row_id, TensorRow *out);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_BASE_OP_H_
