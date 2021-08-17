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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_sampler.h"
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
SubsetSamplerObj::SubsetSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : indices_(std::move(indices)), num_samples_(num_samples) {}

// Destructor
SubsetSamplerObj::~SubsetSamplerObj() = default;

Status SubsetSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("SubsetRandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  return Status::OK();
}

Status SubsetSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SubsetSamplerRT>(indices_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> SubsetSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardSample>(indices_);

  return mind_sampler;
}
#endif
Status SubsetSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SubsetSampler";
  args["indices"] = indices_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status SubsetSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("indices") != json_obj.end(), "Failed to find indices");
  std::vector<int64_t> indices = json_obj["indices"];
  *sampler = std::make_shared<SubsetSamplerObj>(indices, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

std::shared_ptr<SamplerObj> SubsetSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SubsetSamplerObj>(indices_, num_samples_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}
}  // namespace dataset
}  // namespace mindspore
