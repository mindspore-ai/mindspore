/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_DISTRIBUTED_SAMPLER_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_DISTRIBUTED_SAMPLER_IR_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "include/api/status.h"
#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_operator.h"
#endif

namespace mindspore {
namespace dataset {

// Internal Sampler class forward declaration
class SamplerRT;

class DistributedSamplerObj : public SamplerObj {
 public:
  DistributedSamplerObj(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples, uint32_t seed,
                        int64_t offset, bool even_dist);

  ~DistributedSamplerObj();

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

#ifndef ENABLE_ANDROID
  /// \brief Function for read sampler from JSON object
  /// \param[in] json_obj JSON object to be read
  /// \param[in] num_samples number of sample in the sampler
  /// \param[out] sampler Sampler constructed from parameters in JSON object
  /// \return Status of the function
  static Status from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler);
#endif

  Status ValidateParams() override;

  /// \brief Function to get the shard id of sampler
  /// \return The shard id of sampler
  int64_t ShardId() override;

 private:
  int64_t num_shards_;
  int64_t shard_id_;
  bool shuffle_;
  int64_t num_samples_;
  uint32_t seed_;
  int64_t offset_;
  bool even_dist_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_DISTRIBUTED_SAMPLER_IR_H_
