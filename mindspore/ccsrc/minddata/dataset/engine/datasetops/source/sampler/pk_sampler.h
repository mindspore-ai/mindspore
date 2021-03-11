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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_

#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class PKSamplerRT : public SamplerRT {  // NOT YET FINISHED
 public:
  // @param num_samples - the number of samples to draw.  value of 0 means to take the full amount
  // @param int64_t val
  // @param bool shuffle - shuffle all classIds or not, if true, classes may be 5,1,4,3,2
  // @param int64_t samplesPerBuffer - Num of Sampler Ids to fetch via 1 GetNextBuffer call
  PKSamplerRT(int64_t num_samples, int64_t val, bool shuffle,
              int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // default destructor
  ~PKSamplerRT() = default;

  // @param std::unique_ptr<DataBuffer pBuffer
  // @param int32_t workerId
  // @return Status The status code returned
  Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

  // first handshake between leaf source op and Sampler. This func will determine the amount of data
  // in the dataset that we can sample from.
  // @param op - leaf op pointer, pass in so Sampler can ask it about how much data there is
  // @return
  Status HandshakeRandomAccessOp(const RandomAccessOp *op) override;

  // init sampler, to be called by python or Handshake
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return Status The status code returned
  Status ResetSampler() override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief PK cannot return an exact value because num_classes is not known until runtime, hence -1 is used
  /// \param[out] num_rows
  /// \return -1, which means PKSampler doesn't know how much data
  int64_t CalculateNumSamples(int64_t num_rows) override { return -1; }

 private:
  bool shuffle_;
  uint32_t seed_;
  int64_t next_id_;
  int64_t samples_per_class_;
  std::mt19937 rnd_;
  std::vector<int64_t> labels_;
  std::map<int32_t, std::vector<int64_t>> label_to_ids_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_
