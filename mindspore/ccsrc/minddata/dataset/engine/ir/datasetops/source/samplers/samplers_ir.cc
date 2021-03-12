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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"

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
SamplerObj::SamplerObj() {}

Status SamplerObj::BuildChildren(std::shared_ptr<SamplerRT> *const sampler) {
  for (auto child : children_) {
    std::shared_ptr<SamplerRT> sampler_rt = nullptr;
    RETURN_IF_NOT_OK(child->SamplerBuild(&sampler_rt));
    RETURN_IF_NOT_OK((*sampler)->AddChild(sampler_rt));
  }
  return Status::OK();
}

Status SamplerObj::AddChildSampler(std::shared_ptr<SamplerObj> child) {
  if (child == nullptr) {
    return Status::OK();
  }

  // Only samplers can be added, not any other DatasetOp.
  std::shared_ptr<SamplerObj> sampler = std::dynamic_pointer_cast<SamplerObj>(child);
  if (!sampler) {
    RETURN_STATUS_UNEXPECTED("Cannot add child, child is not a sampler object.");
  }

  // Samplers can have at most 1 child.
  if (!children_.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot add child sampler, this sampler already has a child.");
  }

  children_.push_back(child);

  return Status::OK();
}

/* ####################################### Derived Sampler classes ################################# */

// DistributedSampler
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
  *sampler = std::make_shared<dataset::DistributedSamplerRT>(num_samples_, num_shards_, shard_id_, shuffle_, seed_,
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
  args["sampler_name"] = "DistributedSampler";
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  args["shuffle"] = shuffle_;
  args["num_samples"] = num_samples_;
  args["offset"] = offset_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

// PKSampler
PKSamplerObj::PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples)
    : num_val_(num_val), shuffle_(shuffle), num_samples_(num_samples) {}

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
  args["sampler_name"] = "PKSampler";
  args["num_val"] = num_val_;
  args["shuffle"] = shuffle_;
  args["num_samples"] = num_samples_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

Status PKSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::PKSamplerRT>(num_samples_, num_val_, shuffle_);
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

// PreBuiltOperation
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler) : sp_(std::move(sampler)) {}

#ifndef ENABLE_ANDROID
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<mindrecord::ShardOperator> sampler)
    : sp_minddataset_(std::move(sampler)) {}
#endif

Status PreBuiltSamplerObj::ValidateParams() { return Status::OK(); }

Status PreBuiltSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) {
  Status s = BuildChildren(&sp_);
  if (s.IsOk())
    *sampler = sp_;
  else
    *sampler = nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> PreBuiltSamplerObj::BuildForMindDataset() { return sp_minddataset_; }
#endif

std::shared_ptr<SamplerObj> PreBuiltSamplerObj::SamplerCopy() {
#ifndef ENABLE_ANDROID
  if (sp_minddataset_ != nullptr) {
    auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_minddataset_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }
#endif
  auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_);
  for (auto child : children_) {
    sampler->AddChildSampler(child);
  }
  return sampler;
}

Status PreBuiltSamplerObj::to_json(nlohmann::json *const out_json) {
  RETURN_IF_NOT_OK(sp_->to_json(out_json));
  return Status::OK();
}

// RandomSampler
RandomSamplerObj::RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch)
    : replacement_(replacement), num_samples_(num_samples), reshuffle_each_epoch_(reshuffle_each_epoch) {}

Status RandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("RandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status RandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  args["sampler_name"] = "RandomSampler";
  args["replacement"] = replacement_;
  args["num_samples"] = num_samples_;
  args["reshuffle_each_epoch"] = reshuffle_each_epoch_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

Status RandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::RandomSamplerRT>(num_samples_, replacement_, reshuffle_each_epoch_);
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

// SequentialSampler
SequentialSamplerObj::SequentialSamplerObj(int64_t start_index, int64_t num_samples)
    : start_index_(start_index), num_samples_(num_samples) {}

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
  args["sampler_name"] = "SequentialSampler";
  args["start_index"] = start_index_;
  args["num_samples"] = num_samples_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

Status SequentialSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SequentialSamplerRT>(num_samples_, start_index_);
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

// SubsetSampler
SubsetSamplerObj::SubsetSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : indices_(std::move(indices)), num_samples_(num_samples) {}

Status SubsetSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("SubsetRandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  return Status::OK();
}

Status SubsetSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SubsetSamplerRT>(num_samples_, indices_);
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
  args["sampler_name"] = "SubsetSampler";
  args["indices"] = indices_;
  args["num_samples"] = num_samples_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

// SubsetRandomSampler
SubsetRandomSamplerObj::SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : SubsetSamplerObj(std::move(indices), num_samples) {}

Status SubsetRandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SubsetRandomSamplerRT>(num_samples_, indices_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> SubsetRandomSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardSample>(indices_, GetSeed());

  return mind_sampler;
}
#endif

Status SubsetRandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  args["sampler_name"] = "SubsetRandomSampler";
  args["indices"] = indices_;
  args["num_samples"] = num_samples_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

// WeightedRandomSampler
WeightedRandomSamplerObj::WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples, bool replacement)
    : weights_(std::move(weights)), num_samples_(num_samples), replacement_(replacement) {}

Status WeightedRandomSamplerObj::ValidateParams() {
  if (weights_.empty()) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: weights vector must not be empty");
  }
  int32_t zero_elem = 0;
  for (int32_t i = 0; i < weights_.size(); ++i) {
    if (weights_[i] < 0) {
      RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: weights vector must not contain negative number, got: " +
                               std::to_string(weights_[i]));
    }
    if (weights_[i] == 0.0) {
      zero_elem++;
    }
  }
  if (zero_elem == weights_.size()) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: elements of weights vector must not be all zero");
  }
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status WeightedRandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  args["sampler_name"] = "WeightedRandomSampler";
  args["weights"] = weights_;
  args["num_samples"] = num_samples_;
  args["replacement"] = replacement_;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

Status WeightedRandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  *sampler = std::make_shared<dataset::WeightedRandomSamplerRT>(num_samples_, weights_, replacement_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

}  // namespace dataset
}  // namespace mindspore
