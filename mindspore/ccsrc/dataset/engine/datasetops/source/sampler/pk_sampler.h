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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_

#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class PKSampler : public Sampler {  // NOT YET FINISHED
 public:
  // @param int64_t kVal
  // @param bool shuffle - shuffle all classIds or not, if true, classes may be 5,1,4,3,2
  // @param int64_t samplesPerBuffer - Num of Sampler Ids to fetch via 1 GetNextBuffer call
  explicit PKSampler(int64_t val, bool shuffle = false,
                     int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // default destructor
  ~PKSampler() = default;

  // @param std::unique_ptr<DataBuffer pBuffer
  // @param int32_t workerId
  // @return - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) override;

  // first handshake between StorageOp and Sampler
  // @param op - StorageOp pointer, pass in so Sampler can call GetNumSamples() and get ClassIds()
  // @return
  Status Init(const RandomAccessOp *op) override;

  // for next epoch of sampleIds
  // @return - The error code return
  Status Reset() override;

 private:
  bool shuffle_;
  uint32_t seed_;
  int64_t next_id_;
  int64_t num_pk_samples_;
  int64_t samples_per_class_;
  std::mt19937 rnd_;
  std::vector<int64_t> labels_;
  std::map<int32_t, std::vector<int64_t>> label_to_ids_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PK_SAMPLER_H_
