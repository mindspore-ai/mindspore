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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
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

class PreBuiltSamplerObj : public SamplerObj {
 public:
  explicit PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler);
#ifndef ENABLE_ANDROID
  explicit PreBuiltSamplerObj(std::shared_ptr<mindrecord::ShardOperator> sampler);
#endif

  ~PreBuiltSamplerObj();

  Status SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  std::shared_ptr<SamplerObj> SamplerCopy() override;

  Status ValidateParams() override;

  Status to_json(nlohmann::json *const out_json) override;

 private:
  std::shared_ptr<SamplerRT> sp_;
#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> sp_minddataset_;
#endif
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_
