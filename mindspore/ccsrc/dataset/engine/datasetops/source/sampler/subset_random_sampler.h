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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
// Randomly samples elements from a given list of indices, without replacement.
class SubsetRandomSampler : public Sampler {
 public:
  // Constructor.
  // @param num_samples The number of samples to draw. 0 for the full amount.
  // @param indices List of indices from where we will randomly draw samples.
  // @param samples_per_buffer The number of ids we draw on each call to GetNextBuffer().
  // When samplesPerBuffer=0, GetNextBuffer() will draw all the sample ids and return them at once.
  explicit SubsetRandomSampler(int64_t num_samples, const std::vector<int64_t> &indices,
                               std::int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~SubsetRandomSampler() = default;

  // Initialize the sampler.
  // @return Status
  Status InitSampler() override;

  // Reset the internal variable to the initial state and reshuffle the indices.
  // @return Status
  Status ResetSampler() override;

  // Get the sample ids.
  // @param[out] out_buffer The address of a unique_ptr to DataBuffer where the sample ids will be placed.
  // @note the sample ids (int64_t) will be placed in one Tensor and be placed into pBuffer.
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

 private:
  // A list of indices (already randomized in constructor).
  std::vector<int64_t> indices_;

  // Current sample id.
  int64_t sample_id_;

  // Current buffer id.
  int64_t buffer_id_;

  // A random number generator.
  std::mt19937 rand_gen_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_
