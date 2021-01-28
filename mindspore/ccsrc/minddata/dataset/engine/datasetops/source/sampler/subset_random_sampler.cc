/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
// Constructor.
SubsetRandomSamplerRT::SubsetRandomSamplerRT(int64_t num_samples, const std::vector<int64_t> &indices,
                                             int64_t samples_per_buffer)
    : SubsetSamplerRT(num_samples, indices, samples_per_buffer) {}

// Initialized this Sampler.
Status SubsetRandomSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }

  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  // num_samples_ could be smaller than the total number of input id's.
  // We will shuffle the full set of id's, but only select the first num_samples_ of them later.
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return SubsetSamplerRT::InitSampler();
}

// Reset the internal variable to the initial state.
Status SubsetRandomSamplerRT::ResetSampler() {
  // Randomized the indices again.
  rand_gen_.seed(GetSeed());
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return SubsetSamplerRT::ResetSampler();
}

void SubsetRandomSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: SubsetRandomSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status SubsetRandomSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SubsetSamplerRT::to_json(&args));
  args["sampler_name"] = "SubsetRandomSampler";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
