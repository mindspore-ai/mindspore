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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_

#include <limits>
#include <memory>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class SequentialSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param num_samples - The number of samples to draw. A value of 0 indicates the sampler should produce the
  //                      full amount of ids from the dataset
  // @param start_index - The starting index value
  // @param int64_t samplesPerBuffer - Num of Sampler Ids to fetch via 1 GetNextBuffer call
  SequentialSamplerRT(int64_t num_samples, int64_t start_index,
                      int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~SequentialSamplerRT() = default;

  // init sampler, called by python
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return Status The status code returned
  Status ResetSampler() override;

  // Op calls this to get next Buffer that contains all the sampleIds
  // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to corresponding Dataset Op
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

  /// \brief Recursively calls this function on its children to get the actual number of samples on a tree of samplers
  /// \note This is not a getter for num_samples_. For example, if num_samples_ is 0 or if it's smaller than num_rows,
  ///     then num_samples_ is not returned at all.
  /// \param[in] num_rows The total number of rows in the dataset
  /// \return int64_t Calculated number of samples
  int64_t CalculateNumSamples(int64_t num_rows) override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int64_t current_id_;   // The id sequencer.  Each new id increments from this
  int64_t start_index_;  // The starting id.  current_id_ begins from here.
  int64_t id_count_;     // An internal counter that tracks how many ids have been produced
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_
