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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"

namespace mindspore {
namespace dataset {
namespace api {

SamplerObj::SamplerObj() {}

/// Function to create a Distributed Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle,
                                                          int64_t num_samples, uint32_t seed, bool even_dist) {
  auto sampler = std::make_shared<DistributedSamplerObj>(num_shards, shard_id, shuffle, num_samples, seed, even_dist);
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
                                             uint32_t seed, bool even_dist)
    : num_shards_(num_shards),
      shard_id_(shard_id),
      shuffle_(shuffle),
      num_samples_(num_samples),
      seed_(seed),
      even_dist_(even_dist) {}

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

  return true;
}

std::shared_ptr<Sampler> DistributedSamplerObj::Build() {
  return std::make_shared<dataset::DistributedSampler>(num_samples_, num_shards_, shard_id_, shuffle_, seed_,
                                                       even_dist_);
}

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

std::shared_ptr<Sampler> PKSamplerObj::Build() {
  return std::make_shared<dataset::PKSampler>(num_samples_, num_val_, shuffle_);
}

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

std::shared_ptr<Sampler> RandomSamplerObj::Build() {
  bool reshuffle_each_epoch = true;
  auto sampler = std::make_shared<dataset::RandomSampler>(num_samples_, replacement_, reshuffle_each_epoch);
  return sampler;
}

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

std::shared_ptr<Sampler> SequentialSamplerObj::Build() {
  auto sampler = std::make_shared<dataset::SequentialSampler>(num_samples_, start_index_);
  return sampler;
}

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

std::shared_ptr<Sampler> SubsetRandomSamplerObj::Build() {
  auto sampler = std::make_shared<dataset::SubsetRandomSampler>(num_samples_, indices_);
  return sampler;
}

// WeightedRandomSampler
WeightedRandomSamplerObj::WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples, bool replacement)
    : weights_(std::move(weights)), num_samples_(num_samples), replacement_(replacement) {}

bool WeightedRandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "WeightedRandomSampler: invalid num_samples: " << num_samples_;
    return false;
  }
  return true;
}

std::shared_ptr<Sampler> WeightedRandomSamplerObj::Build() {
  auto sampler = std::make_shared<dataset::WeightedRandomSampler>(num_samples_, weights_, replacement_);
  return sampler;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
