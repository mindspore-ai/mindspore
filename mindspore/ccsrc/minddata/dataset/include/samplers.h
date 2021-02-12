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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_

#include <memory>
#include <vector>

// FIXME - This internal IR header will be removed when external API classes are provided
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

namespace mindspore {
namespace dataset {

class DistributedSamplerObj;
class PKSamplerObj;
class PreBuiltSamplerObj;
class RandomSamplerObj;
class SequentialSamplerObj;
class SubsetSamplerObj;
class SubsetRandomSamplerObj;
class WeightedRandomSamplerObj;

/// Function to create a Distributed Sampler.
/// \notes A Sampler that access a shard of the dataset.
/// \param[in] num_shards - Number of shards to divide the dataset into.
/// \param[in] shard_id - Shard ID of the current shard within num_shards.
/// \param[in] shuffle - If true, the indices are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] seed - The seed in use when shuffle is true.
/// \param[in] offset - The starting position where access to elements in the dataset begins.
/// \param[in] even_dist - If true, each shard would return the same number of rows (default to true).
///     If false the total rows returned by all the shards would not have overlap.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle = true,
                                                          int64_t num_samples = 0, uint32_t seed = 1,
                                                          int64_t offset = -1, bool even_dist = true);

/// Function to create a PK Sampler.
/// \notes Samples K elements for each P class in the dataset.
///        This will sample all classes.
/// \param[in] num_val - Number of elements to sample for each class.
/// \param[in] shuffle - If true, the class IDs are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<PKSamplerObj> PKSampler(int64_t num_val, bool shuffle = false, int64_t num_samples = 0);

/// Function to create a Random Sampler.
/// \notes Samples the elements randomly.
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<RandomSamplerObj> RandomSampler(bool replacement = false, int64_t num_samples = 0);

/// Function to create a Sequential Sampler.
/// \notes Samples the dataset elements sequentially, same as not having a sampler.
/// \param[in] start_index - Index to start sampling at (default to start at first id).
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SequentialSamplerObj> SequentialSampler(int64_t start_index = 0, int64_t num_samples = 0);

/// Function to create a Subset  Sampler.
/// \notes Samples the elements from a sequence of indices.
/// \param[in] indices - A vector sequence of indices.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SubsetSamplerObj> SubsetSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

/// Function to create a Subset Random Sampler.
/// \notes Samples the elements randomly from a sequence of indices.
/// \param[in] indices - A vector sequence of indices.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SubsetRandomSamplerObj> SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

/// Function to create a Weighted Random Sampler.
/// \notes Samples the elements from [0, len(weights) - 1] randomly with the given
///        weights (probabilities).
/// \param[in] weights - A vector sequence of weights, not necessarily summing up to 1.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<WeightedRandomSamplerObj> WeightedRandomSampler(std::vector<double> weights, int64_t num_samples = 0,
                                                                bool replacement = true);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
