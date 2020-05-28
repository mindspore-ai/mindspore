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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class RandomSampler : public Sampler {
 public:
  // Constructor
  // @param bool replacement - put he id back / or not after a sample
  // @param int64_t numSamples - number samples to draw
  // @param int64_t samplesPerBuffer - Num of Sampler Ids to fetch via 1 GetNextBuffer call
  explicit RandomSampler(bool replacement = false, bool reshuffle_each_epoch = true,
                         int64_t num_samples = std::numeric_limits<int64_t>::max(),
                         int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~RandomSampler() = default;

  // Op calls this to get next Buffer that contains all the sampleIds
  // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to StorageOp
  // @param int32_t workerId - not meant to be used
  // @return - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) override;

  // meant to be called by base class or python
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return - The error code return
  Status Reset() override;

  virtual void Print(std::ostream &out, bool show_all) const;

 private:
  uint32_t seed_;
  bool replacement_;
  int64_t user_num_samples_;
  std::vector<int64_t> shuffled_ids_;  // only used for NO REPLACEMENT
  int64_t next_id_;
  std::mt19937 rnd_;
  std::unique_ptr<std::uniform_int_distribution<int64_t>> dist;
  bool reshuffle_each_epoch_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
