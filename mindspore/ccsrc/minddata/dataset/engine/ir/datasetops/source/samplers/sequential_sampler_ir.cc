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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
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
SequentialSamplerObj::SequentialSamplerObj(int64_t start_index, int64_t num_samples)
    : start_index_(start_index), num_samples_(num_samples) {}

// Destructor
SequentialSamplerObj::~SequentialSamplerObj() = default;

Status SequentialSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("SequentialSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  if (start_index_ < 0) {
    RETURN_STATUS_UNEXPECTED("SequentialSampler: start_index_ must be greater than or equal to 0, but got: " +
                             std::to_string(start_index_));
  }

  return Status::OK();
}

Status SequentialSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SequentialSampler";
  args["start_index"] = start_index_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status SequentialSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples,
                                       std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("start_index") != json_obj.end(), "Failed to find start_index");
  int64_t start_index = json_obj["start_index"];
  *sampler = std::make_shared<SequentialSamplerObj>(start_index, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

Status SequentialSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SequentialSamplerRT>(start_index_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> SequentialSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardSequentialSample>(num_samples_, start_index_);

  return mind_sampler;
}
#endif
std::shared_ptr<SamplerObj> SequentialSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SequentialSamplerObj>(start_index_, num_samples_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}
}  // namespace dataset
}  // namespace mindspore
