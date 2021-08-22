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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/core/config_manager.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/util/random.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_sequential_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#endif

namespace mindspore {
namespace dataset {
// Constructor
RandomSamplerObj::RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch)
    : replacement_(replacement), num_samples_(num_samples), reshuffle_each_epoch_(reshuffle_each_epoch) {}

// Destructor
RandomSamplerObj::~RandomSamplerObj() = default;

Status RandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("RandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status RandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "RandomSampler";
  args["replacement"] = replacement_;
  args["reshuffle_each_epoch"] = reshuffle_each_epoch_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status RandomSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("replacement") != json_obj.end(), "Failed to find replacement");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("reshuffle_each_epoch") != json_obj.end(),
                               "Failed to find reshuffle_each_epoch");
  bool replacement = json_obj["replacement"];
  bool reshuffle_each_epoch = json_obj["reshuffle_each_epoch"];
  *sampler = std::make_shared<RandomSamplerObj>(replacement, num_samples, reshuffle_each_epoch);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

Status RandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::RandomSamplerRT>(replacement_, num_samples_, reshuffle_each_epoch_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> RandomSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler =
    std::make_shared<mindrecord::ShardShuffle>(GetSeed(), num_samples_, replacement_, reshuffle_each_epoch_);

  return mind_sampler;
}
#endif

std::shared_ptr<SamplerObj> RandomSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<RandomSamplerObj>(replacement_, num_samples_, reshuffle_each_epoch_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}
}  // namespace dataset
}  // namespace mindspore
