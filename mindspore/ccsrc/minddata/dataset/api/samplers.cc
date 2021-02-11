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

#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

namespace mindspore {
namespace dataset {

/// Function to create a Distributed Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle,
                                                          int64_t num_samples, uint32_t seed, int64_t offset,
                                                          bool even_dist) {
  auto sampler =
    std::make_shared<DistributedSamplerObj>(num_shards, shard_id, shuffle, num_samples, seed, offset, even_dist);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a PK Sampler.
std::shared_ptr<PKSamplerObj> PKSampler(int64_t num_val, bool shuffle, int64_t num_samples) {
  auto sampler = std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Random Sampler.
std::shared_ptr<RandomSamplerObj> RandomSampler(bool replacement, int64_t num_samples) {
  auto sampler = std::make_shared<RandomSamplerObj>(replacement, num_samples);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Sequential Sampler.
std::shared_ptr<SequentialSamplerObj> SequentialSampler(int64_t start_index, int64_t num_samples) {
  auto sampler = std::make_shared<SequentialSamplerObj>(start_index, num_samples);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Subset Random Sampler.
std::shared_ptr<SubsetSamplerObj> SubsetSampler(std::vector<int64_t> indices, int64_t num_samples) {
  auto sampler = std::make_shared<SubsetSamplerObj>(std::move(indices), num_samples);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Subset Random Sampler.
std::shared_ptr<SubsetRandomSamplerObj> SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples) {
  auto sampler = std::make_shared<SubsetRandomSamplerObj>(std::move(indices), num_samples);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

/// Function to create a Weighted Random Sampler.
std::shared_ptr<WeightedRandomSamplerObj> WeightedRandomSampler(std::vector<double> weights, int64_t num_samples,
                                                                bool replacement) {
  auto sampler = std::make_shared<WeightedRandomSamplerObj>(std::move(weights), num_samples, replacement);
  // Input validation
  if (sampler->ValidateParams().IsError()) {
    return nullptr;
  }
  return sampler;
}

}  // namespace dataset
}  // namespace mindspore
