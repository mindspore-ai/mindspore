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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_sampler.h"

namespace mindspore {
namespace dataset {
/// Randomly samples elements from a given list of indices, without replacement.
class SubsetRandomSamplerRT : public SubsetSamplerRT {
 public:
  /// Constructor.
  /// \param indices List of indices from where we will randomly draw samples.
  /// \param num_samples The number of samples to draw. 0 for the full amount.
  /// \param samples_per_tensor The number of ids we draw on each call to GetNextSample().
  /// When samples_per_tensor=0, GetNextSample() will draw all the sample ids and return them at once.
  SubsetRandomSamplerRT(const std::vector<int64_t> &indices, int64_t num_samples,
                        std::int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  /// Destructor.
  ~SubsetRandomSamplerRT() = default;

  /// Initialize the sampler.
  /// \return Status
  Status InitSampler() override;

  /// Reset the internal variable to the initial state and reshuffle the indices.
  /// \return Status
  Status ResetSampler() override;

  /// Printer for debugging purposes.
  /// \param out - output stream to write to
  /// \param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  // A random number generator.
  std::mt19937 rand_gen_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SUBSET_RANDOM_SAMPLER_H_
