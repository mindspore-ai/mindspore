/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_MINDRECORD_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_MINDRECORD_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/mindrecord/include/shard_reader.h"

namespace mindspore {
namespace dataset {
class MindRecordSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param shard_reader - shard_reader
  // @param int64_t samples_per_tensor - Num of Sampler Ids to fetch via 1 GetNextSample call
  MindRecordSamplerRT(mindrecord::ShardReader *shard_reader,
                      int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~MindRecordSamplerRT() = default;

  // Op calls this to get next set of sampleIds
  // @param out - Tensor of sample ids to be returned to caller
  // @return Status The status code returned
  Status GetNextSample(TensorRow *out) override;

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
  mindrecord::ShardReader *shard_reader_;  // back pointer to the shard reader
  const std::vector<int> *sample_ids_;     // read-only back pointer into mind record sampler ids
  int64_t next_id_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_MINDRECORD_SAMPLER_H_
