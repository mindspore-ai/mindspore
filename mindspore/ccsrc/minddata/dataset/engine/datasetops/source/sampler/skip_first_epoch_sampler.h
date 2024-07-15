/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_

#include <limits>

#include <nlohmann/json.hpp>

#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

namespace mindspore {
namespace dataset {
class SkipFirstEpochSamplerRT : public SequentialSamplerRT {
 public:
  // Constructor
  SkipFirstEpochSamplerRT(int64_t start_index, int64_t num_samples,
                          int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~SkipFirstEpochSamplerRT() override = default;

  Status GetNextSample(TensorRow *out) override;

  /// \brief Reset for next epoch.
  /// \param[in] failover_reset A boolean to show whether we are resetting the pipeline
  /// \return Status The status code returned
  Status ResetSampler(const bool failover_reset = false) override;

  /// \brief Gets the number of samples available
  /// \note Since this sampler returns different number of samples in the first epoch (compared to other epochs), this
  ///     function always returns -1
  /// \param[in] num_rows The total number of rows in the dataset
  /// \return int64_t Calculated number of samples (always -1)
  int64_t CalculateNumSamples(const int64_t num_rows) override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int64_t sample_need_to_skip_;
  bool first_epoch_done_ = false;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_
