/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_

#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class DistributedSampler : public Sampler {
 public:
  /// \brief Constructor
  /// \param[in] num_samples The total number of rows in the dataset
  /// \param[in] num_dev Total number of shards for the distributed sampler
  /// \param[in] dev_id Device id of the shard
  /// \param[in] shuffle Option to shuffle
  /// \param seed Seed parameter to shuffle, default to max unsigned int (different seed in sampler will
  ///     result in different samples being picked
  /// \param[in] offset The starting position which the elements in the dataset are send to.The application
  ///     scenario of this parameter is when the concatdataset is set distributedSampler
  /// \param even_dist The option to indicate whether or not each shard returns the same number of rows.
  ///     This option is not exposed in the python API. Current behavior is that the remainder will always
  ///     be handled by the first n shards, n being the corresponding device id.
  DistributedSampler(int64_t num_samples, int64_t num_dev, int64_t dev_id, bool shuffle,
                     uint32_t seed = std::numeric_limits<uint32_t>::max(), int64_t offset = -1, bool even_dist = true);

  /// \brief default destructor
  ~DistributedSampler() = default;

  /// \param std::unique_ptr<DataBuffer> * pBuffer
  /// \param int32_t workerId
  /// \return Status code
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

  /// Init sampler, called by base class or python
  Status InitSampler() override;

  /// \brief for next epoch of sampleIds
  /// \return Status code
  Status ResetSampler() override;

  int64_t GetDeviceID() { return device_id_; }

  int64_t GetDeviceNum() { return num_devices_; }

  void Print(std::ostream &out, bool show_all) const override;

 private:
  int64_t cnt_;  // number of samples that have already been filled in to buffer
  uint32_t seed_;
  int64_t device_id_;
  int64_t num_devices_;
  bool shuffle_;
  std::mt19937 rnd_;
  std::vector<int64_t> shuffle_vec_;
  bool even_dist_;
  int64_t offset_;
  bool non_empty_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_
