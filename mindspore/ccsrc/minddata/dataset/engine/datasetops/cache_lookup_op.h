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
  class Builder {
   public:
    /// \brief Builder constructor. Creates the builder object.
    /// \note No default args
    Builder();

    /// Default destructor
    ~Builder() = default;

    /// Setter method.
    /// \treturn Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      build_num_workers_ = num_workers;
      return *this;
    }

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t connector_size) {
      build_op_connector_size_ = connector_size;
      return *this;
    }

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetClient(std::shared_ptr<CacheClient> cache_client) {
      build_cache_client_ = cache_client;
      return *this;
    }

    /// \brief Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      build_sampler_ = std::move(sampler);
      return *this;
    }

    /// \brief The builder "build" method creates the final object and does some init on it.
    /// \param ptr The shared_ptr to the new CacheLookupOp object
    /// \return Status
    Status Build(std::shared_ptr<CacheLookupOp> *ptr);

   private:
    int32_t build_num_workers_;
    int32_t rows_per_buffer_;
    int32_t build_op_connector_size_;
    std::shared_ptr<CacheClient> build_cache_client_;
    std::shared_ptr<SamplerRT> build_sampler_;

    // Check if the required parameters are set by the builder.
    // \return Status The status code returned
    Status SanityCheck() const;
  };
  /// \brief Constructor
  /// \note It takes the same argument as the base class.
  /// \see CacheBase
  CacheLookupOp(int32_t num_workers, int32_t op_connector_size, int32_t rows_per_buf,
                std::shared_ptr<CacheClient> cache_client, std::shared_ptr<SamplerRT> sampler)
      : CacheBase(num_workers, op_connector_size, rows_per_buf, cache_client, sampler), SamplerRT(*(sampler.get())) {}
  ~CacheLookupOp() = default;
  // As a parallel op, we override these two functions
  Status operator()() override;
  Status WorkerEntry(int32_t worker_id) override;
  // As a sampler, we override the following functions
  Status ResetSampler() override;
  Status HandshakeRandomAccessOp(const RandomAccessOp *op) override;
  Status InitSampler() override;
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;
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
