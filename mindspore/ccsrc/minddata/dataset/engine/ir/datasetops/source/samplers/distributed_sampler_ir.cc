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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/distributed_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
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
DistributedSamplerObj::DistributedSamplerObj(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples,
                                             uint32_t seed, int64_t offset, bool even_dist)
    : num_shards_(num_shards),
      shard_id_(shard_id),
      shuffle_(shuffle),
      num_samples_(num_samples),
      seed_(seed),
      offset_(offset),
      even_dist_(even_dist) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

// Destructor
DistributedSamplerObj::~DistributedSamplerObj() = default;

Status DistributedSamplerObj::ValidateParams() {
  if (num_shards_ <= 0) {
    RETURN_STATUS_UNEXPECTED("DistributedSampler: num_shards must be greater than 0, but got: " +
                             std::to_string(num_shards_));
  }

  if (shard_id_ < 0 || shard_id_ >= num_shards_) {
    RETURN_STATUS_UNEXPECTED("DistributedSampler: shard_id must be in range [0, " + std::to_string(num_shards_) +
                             "), but got: " + std::to_string(shard_id_));
  }

  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("DistributedSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  if (offset_ > num_shards_) {
    RETURN_STATUS_UNEXPECTED("DistributedSampler: offset must be no more than num_shards(" +
                             std::to_string(num_shards_) + "), but got: " + std::to_string(offset_));
  }

  return Status::OK();
}

Status DistributedSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::DistributedSamplerRT>(num_shards_, shard_id_, shuffle_, num_samples_, seed_,
                                                             offset_, even_dist_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> DistributedSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardDistributedSample>(num_shards_, shard_id_, shuffle_, seed_,
                                                                           num_samples_, offset_);
  return mind_sampler;
}
#endif

Status DistributedSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "DistributedSampler";
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  args["shuffle"] = shuffle_;
  args["seed"] = seed_;
  args["offset"] = offset_;
  args["num_samples"] = num_samples_;
  args["even_dist"] = even_dist_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status DistributedSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples,
                                        std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Failed to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Failed to find shard_id");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Failed to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("seed") != json_obj.end(), "Failed to find seed");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("offset") != json_obj.end(), "Failed to find offset");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("even_dist") != json_obj.end(), "Failed to find even_dist");
  int64_t num_shards = json_obj["num_shards"];
  int64_t shard_id = json_obj["shard_id"];
  bool shuffle = json_obj["shuffle"];
  uint32_t seed = json_obj["seed"];
  int64_t offset = json_obj["offset"];
  bool even_dist = json_obj["even_dist"];
  *sampler =
    std::make_shared<DistributedSamplerObj>(num_shards, shard_id, shuffle, num_samples, seed, offset, even_dist);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

std::shared_ptr<SamplerObj> DistributedSamplerObj::SamplerCopy() {
  auto sampler =
    std::make_shared<DistributedSamplerObj>(num_shards_, shard_id_, shuffle_, num_samples_, seed_, offset_, even_dist_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}

int64_t DistributedSamplerObj::ShardId() { return shard_id_; }
}  // namespace dataset
}  // namespace mindspore
