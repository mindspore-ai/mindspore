/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"

#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_sequential_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "minddata/dataset/util/random.h"
#endif

namespace mindspore {
namespace dataset {

#define RETURN_NULL_IF_ERROR(_s) \
  do {                           \
    Status __rc = (_s);          \
    if (__rc.IsError()) {        \
      MS_LOG(ERROR) << __rc;     \
      return nullptr;            \
    }                            \
  } while (false)

// Constructor
SamplerObj::SamplerObj() {}

/// Function to create a Distributed Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle,
                                                          int64_t num_samples, uint32_t seed, int64_t offset,
                                                          bool even_dist) {
  auto sampler =
    std::make_shared<DistributedSamplerObj>(num_shards, shard_id, shuffle, num_samples, seed, offset, even_dist);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a PK Sampler.
std::shared_ptr<PKSamplerObj> PKSampler(int64_t num_val, bool shuffle, int64_t num_samples) {
  auto sampler = std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Random Sampler.
std::shared_ptr<RandomSamplerObj> RandomSampler(bool replacement, int64_t num_samples) {
  auto sampler = std::make_shared<RandomSamplerObj>(replacement, num_samples);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Sequential Sampler.
std::shared_ptr<SequentialSamplerObj> SequentialSampler(int64_t start_index, int64_t num_samples) {
  auto sampler = std::make_shared<SequentialSamplerObj>(start_index, num_samples);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Subset Random Sampler.
std::shared_ptr<SubsetRandomSamplerObj> SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples) {
  auto sampler = std::make_shared<SubsetRandomSamplerObj>(std::move(indices), num_samples);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Weighted Random Sampler.
std::shared_ptr<WeightedRandomSamplerObj> WeightedRandomSampler(std::vector<double> weights, int64_t num_samples,
                                                                bool replacement) {
  auto sampler = std::make_shared<WeightedRandomSamplerObj>(std::move(weights), num_samples, replacement);
  // Input validation
  if (!sampler->ValidateParams()) {
    return nullptr;
  }
  return sampler;
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

bool DistributedSamplerObj::ValidateParams() {
  if (num_shards_ <= 0) {
    MS_LOG(ERROR) << "DistributedSampler: invalid num_shards: " << num_shards_;
    return false;
  }

  if (shard_id_ < 0 || shard_id_ >= num_shards_) {
    MS_LOG(ERROR) << "DistributedSampler: invalid input, shard_id: " << shard_id_ << ", num_shards: " << num_shards_;
    return false;
  }

  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "DistributedSampler: invalid num_samples: " << num_samples_;
    return false;
  }

  if (offset_ > num_shards_) {
    MS_LOG(ERROR) << "DistributedSampler: invalid offset: " << offset_
                  << ", which should be no more than num_shards: " << num_shards_;
    return false;
  }

  return true;
}

std::shared_ptr<SamplerRT> DistributedSamplerObj::Build() {
  // runtime sampler object
  auto sampler = std::make_shared<dataset::DistributedSamplerRT>(num_samples_, num_shards_, shard_id_, shuffle_, seed_,
                                                                 offset_, even_dist_);
  return sampler;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> DistributedSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardDistributedSample>(num_shards_, shard_id_, shuffle_, seed_,
                                                                           num_samples_, offset_);
  return mind_sampler;
}
#endif

// PKSampler
PKSamplerObj::PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples)
    : num_val_(num_val), shuffle_(shuffle), num_samples_(num_samples) {}

bool PKSamplerObj::ValidateParams() {
  if (num_val_ <= 0) {
    MS_LOG(ERROR) << "PKSampler: invalid num_val: " << num_val_;
    return false;
  }

  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "PKSampler: invalid num_samples: " << num_samples_;
    return false;
  }
  return true;
}

std::shared_ptr<SamplerRT> PKSamplerObj::Build() {
  // runtime sampler object
  auto sampler = std::make_shared<dataset::PKSamplerRT>(num_samples_, num_val_, shuffle_);

  return sampler;
}

// PreBuiltOperation
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler) : sp_(std::move(sampler)) {}

#ifndef ENABLE_ANDROID
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<mindrecord::ShardOperator> sampler)
    : sp_minddataset_(std::move(sampler)) {}
#endif

bool PreBuiltSamplerObj::ValidateParams() { return true; }

std::shared_ptr<SamplerRT> PreBuiltSamplerObj::Build() { return sp_; }

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> PreBuiltSamplerObj::BuildForMindDataset() { return sp_minddataset_; }
#endif

std::shared_ptr<SamplerObj> PreBuiltSamplerObj::Copy() {
#ifndef ENABLE_ANDROID
  if (sp_minddataset_ != nullptr) return std::make_shared<PreBuiltSamplerObj>(sp_minddataset_);
#endif
  return std::make_shared<PreBuiltSamplerObj>(sp_);
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

// RandomSampler
RandomSamplerObj::RandomSamplerObj(bool replacement, int64_t num_samples)
    : replacement_(replacement), num_samples_(num_samples) {}

bool RandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "RandomSampler: invalid num_samples: " << num_samples_;
    return false;
  }
  return true;
}

std::shared_ptr<SamplerRT> RandomSamplerObj::Build() {
  // runtime sampler object
  bool reshuffle_each_epoch = true;
  auto sampler = std::make_shared<dataset::RandomSamplerRT>(num_samples_, replacement_, reshuffle_each_epoch);

  return sampler;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> RandomSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  bool reshuffle_each_epoch_ = true;
  auto mind_sampler =
    std::make_shared<mindrecord::ShardShuffle>(GetSeed(), num_samples_, replacement_, reshuffle_each_epoch_);

  return mind_sampler;
}
#endif

// SequentialSampler
SequentialSamplerObj::SequentialSamplerObj(int64_t start_index, int64_t num_samples)
    : start_index_(start_index), num_samples_(num_samples) {}

bool SequentialSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "SequentialSampler: invalid num_samples: " << num_samples_;
    return false;
  }

  if (start_index_ < 0) {
    MS_LOG(ERROR) << "SequentialSampler: invalid start_index: " << start_index_;
    return false;
  }

  return true;
}

std::shared_ptr<SamplerRT> SequentialSamplerObj::Build() {
  // runtime sampler object
  auto sampler = std::make_shared<dataset::SequentialSamplerRT>(num_samples_, start_index_);

  return sampler;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> SequentialSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardSequentialSample>(num_samples_, start_index_);

  return mind_sampler;
}
#endif

// SubsetRandomSampler
SubsetRandomSamplerObj::SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : indices_(std::move(indices)), num_samples_(num_samples) {}

bool SubsetRandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "SubsetRandomSampler: invalid num_samples: " << num_samples_;
    return false;
  }

  return true;
}

std::shared_ptr<SamplerRT> SubsetRandomSamplerObj::Build() {
  // runtime sampler object
  auto sampler = std::make_shared<dataset::SubsetRandomSamplerRT>(num_samples_, indices_);

  return sampler;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> SubsetRandomSamplerObj::BuildForMindDataset() {
  // runtime mindrecord sampler object
  auto mind_sampler = std::make_shared<mindrecord::ShardSample>(indices_, GetSeed());

  return mind_sampler;
}
#endif

// WeightedRandomSampler
WeightedRandomSamplerObj::WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples, bool replacement)
    : weights_(std::move(weights)), num_samples_(num_samples), replacement_(replacement) {}

bool WeightedRandomSamplerObj::ValidateParams() {
  if (weights_.empty()) {
    MS_LOG(ERROR) << "WeightedRandomSampler: weights vector must not be empty";
    return false;
  }
  int32_t zero_elem = 0;
  for (int32_t i = 0; i < weights_.size(); ++i) {
    if (weights_[i] < 0) {
      MS_LOG(ERROR) << "WeightedRandomSampler: weights vector must not contain negative number, got: " << weights_[i];
      return false;
    }
    if (weights_[i] == 0.0) {
      zero_elem++;
    }
  }
  if (zero_elem == weights_.size()) {
    MS_LOG(ERROR) << "WeightedRandomSampler: elements of weights vector must not be all zero";
    return false;
  }
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "WeightedRandomSampler: invalid num_samples: " << num_samples_;
    return false;
  }
  return true;
}

std::shared_ptr<SamplerRT> WeightedRandomSamplerObj::Build() {
  auto sampler = std::make_shared<dataset::WeightedRandomSamplerRT>(num_samples_, weights_, replacement_);
  return sampler;
}

}  // namespace dataset
}  // namespace mindspore
