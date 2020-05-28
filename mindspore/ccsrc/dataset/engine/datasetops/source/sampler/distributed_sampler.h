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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_

#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class DistributedSampler : public Sampler {
 public:
  // @param int64_t numDev
  // @param int64_t devId
  // @param bool shuffle
  DistributedSampler(int64_t num_dev, int64_t dev_id, bool shuffle = true,
                     uint32_t seed = std::numeric_limits<uint32_t>::max());

  // default destructor
  ~DistributedSampler() = default;

  // @param std::unique_ptr<DataBuffer> * pBuffer
  // @param int32_t workerId
  // @return - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) override;

  // Init sampler, called by base class or python
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return - The error code return
  Status Reset() override;

  void Print(std::ostream &out, bool show_all) const override;

 private:
  int64_t cnt_;  // number of samples that have already been filled in to buffer
  uint32_t seed_;
  int64_t device_id_;
  int64_t num_devices_;
  bool shuffle_;
  std::mt19937 rnd_;
  std::vector<int64_t> shuffle_vec_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_DISTRIBUTED_SAMPLER_H_
