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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_LOOKUP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_LOOKUP_OP_H_

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/datasetops/cache_base_op.h"

namespace mindspore {
namespace dataset {
/// \brief provides a memory/disk cache that acts as a save-point within a mappable dataset.
/// \note For non-mappable dataset, please see CacheOp
/// \see CacheOp
class CacheLookupOp : public CacheBase, public SamplerRT {
 public:
  /// \brief Constructor
  /// \note It takes the same argument as the base class.
  /// \see CacheBase
  CacheLookupOp(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<CacheClient> cache_client,
                std::shared_ptr<SamplerRT> sampler)
      : CacheBase(num_workers, op_connector_size, cache_client, sampler), SamplerRT(*(sampler.get())) {}
  ~CacheLookupOp() = default;
  // As a parallel op, we override these two functions
  Status operator()() override;
  Status WorkerEntry(int32_t worker_id) override;
  // As a sampler, we override the following functions
  Status ResetSampler(const bool failover_reset) override;
  Status HandshakeRandomAccessOp(const RandomAccessOp *op, const int32_t reset_count) override;
  Status InitSampler() override;
  Status GetNextSample(TensorRow *out) override;
  void Print(std::ostream &out, bool show_all) const override;
  void SamplerPrint(std::ostream &out, bool show_all) const override;
  bool AllowCacheMiss() override { return true; }
  std::string Name() const override { return kCacheLookupOp; }

 protected:
  Status ComputeColMap() override;

 private:
  WaitPost leaf_op_wp_;

  Status RegisterResources() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_LOOKUP_OP_H_
