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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class RandomSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param int64_t num_samples - number samples to draw
  // @param bool replacement - put he id back / or not after a sample
  // @param reshuffle_each_epoch - T/F to reshuffle after epoch
  // @param int64_t samples_per_buffer - Num of Sampler Ids to fetch via 1 GetNextBuffer call
  RandomSamplerRT(int64_t num_samples, bool replacement, bool reshuffle_each_epoch,
                  int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~RandomSamplerRT() = default;

  // Op calls this to get next Buffer that contains all the sampleIds
  // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to StorageOp
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

  // meant to be called by base class or python
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return Status The status code returned
  Status ResetSampler() override;

  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  uint32_t seed_;
  bool replacement_;
  std::vector<int64_t> shuffled_ids_;  // only used for NO REPLACEMENT
  int64_t next_id_;
  std::mt19937 rnd_;
  std::unique_ptr<std::uniform_int_distribution<int64_t>> dist;
  bool reshuffle_each_epoch_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
