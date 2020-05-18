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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_SAMPLER_H_

#include <memory>
#include <vector>

#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {

class SubsetSampler : public Sampler {
 public:
  // Constructor.
  // @param start_index The index we start sampling from.
  explicit SubsetSampler(int64_t start_index, int64_t subset_size);

  // Destructor.
  ~SubsetSampler() = default;

  // Initialize the sampler.
  // @return Status
  Status InitSampler() override;

  // Reset the internal variable to the initial state and reshuffle the indices.
  // @return Status
  Status Reset() override;

  // Get the sample ids.
  // @param[out] out_buffer The address of a unique_ptr to DataBuffer where the sample ids will be placed.
  // @note the sample ids (int64_t) will be placed in one Tensor.
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) override;

 private:
  int64_t start_index_;
  int64_t subset_size_;
  int64_t current_id_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_SAMPLER_H_
