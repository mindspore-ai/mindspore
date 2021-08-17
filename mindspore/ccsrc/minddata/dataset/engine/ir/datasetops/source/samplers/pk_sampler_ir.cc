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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
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
PKSamplerObj::PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples)
    : num_val_(num_val), shuffle_(shuffle), num_samples_(num_samples) {}

// Destructor
PKSamplerObj::~PKSamplerObj() = default;

Status PKSamplerObj::ValidateParams() {
  if (num_val_ <= 0) {
    RETURN_STATUS_UNEXPECTED("PKSampler: num_val must be greater than 0, but got: " + std::to_string(num_val_));
  }

  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("PKSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status PKSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "PKSampler";
  args["num_val"] = num_val_;
  args["shuffle"] = shuffle_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status PKSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_val") != json_obj.end(), "Failed to find num_val");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Failed to find shuffle");
  int64_t num_val = json_obj["num_val"];
  bool shuffle = json_obj["shuffle"];
  *sampler = std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

Status PKSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::PKSamplerRT>(num_val_, shuffle_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> PKSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  std::shared_ptr<mindrecord::ShardOperator> mind_sampler;
  if (shuffle_ == true) {
    mind_sampler = std::make_shared<mindrecord::ShardPkSample>("label", num_val_, std::numeric_limits<int64_t>::max(),
                                                               GetSeed(), num_samples_);
  } else {
    mind_sampler = std::make_shared<mindrecord::ShardPkSample>("label", num_val_, num_samples_);
  }

  return mind_sampler;
}
#endif

std::shared_ptr<SamplerObj> PKSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<PKSamplerObj>(num_val_, shuffle_, num_samples_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}
}  // namespace dataset
}  // namespace mindspore
