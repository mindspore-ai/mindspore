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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_

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

class RandomSamplerObj : public SamplerObj {
 public:
  RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch = true);

  ~RandomSamplerObj();

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

 private:
  bool replacement_;
  int64_t num_samples_;
  bool reshuffle_each_epoch_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_
