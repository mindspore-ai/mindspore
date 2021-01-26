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
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"

#include <algorithm>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
//  Constructor.
WeightedRandomSamplerRT::WeightedRandomSamplerRT(int64_t num_samples, const std::vector<double> &weights,
                                                 bool replacement, int64_t samples_per_buffer)
    : SamplerRT(num_samples, samples_per_buffer),
      weights_(weights),
      replacement_(replacement),
      sample_id_(0),
      buffer_id_(0) {}

// Initialized this Sampler.
Status WeightedRandomSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0 && num_samples_,
    "Invalid parameter, num_samples and num_rows must be greater than 0, but got num_rows: " +
      std::to_string(num_rows_) + ", num_samples: " + std::to_string(num_samples_));
  CHECK_FAIL_RETURN_UNEXPECTED(samples_per_buffer_ > 0,
                               "Invalid parameter, samples_per_buffer must be greater than 0, but got " +
                                 std::to_string(samples_per_buffer_) + ".\n");

  if (weights_.size() > static_cast<size_t>(num_rows_)) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid parameter, size of sample weights must be less than or equal to num of data, "
                  "otherwise might cause generated id out of bound or other errors, but got weight size: " +
                    std::to_string(weights_.size()) + ", num of data: " + std::to_string(num_rows_));
  }
  if (!replacement_ && (weights_.size() < static_cast<size_t>(num_samples_))) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, without replacement, weight size must be greater than or equal to num_samples, "
      "but got weight size: " +
      std::to_string(weights_.size()) + ", num_samples: " + std::to_string(num_samples_));
  }

  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  samples_per_buffer_ = (samples_per_buffer_ > num_samples_) ? num_samples_ : samples_per_buffer_;

  if (!replacement_) {
    exp_dist_ = std::make_unique<std::exponential_distribution<>>(1);
    InitOnePassSampling();
  } else {
    discrete_dist_ = std::make_unique<std::discrete_distribution<int64_t>>(weights_.begin(), weights_.end());
  }

  is_initialized = true;
  return Status::OK();
}

// Initialized the computation for generating weighted random numbers without replacement using onepass method.
void WeightedRandomSamplerRT::InitOnePassSampling() {
  exp_dist_->reset();
  onepass_ids_.clear();
  std::vector<std::pair<double, int64_t>> val_idx;
  for (size_t i = 0; i < weights_.size(); i++) {
    val_idx.emplace_back(std::make_pair((*exp_dist_)(rand_gen_) / weights_[i], i));
  }

  // Partial sort the first `numSamples` elements.
  std::partial_sort(val_idx.begin(), val_idx.begin() + num_samples_, val_idx.end());
  for (int64_t i = 0; i < num_samples_; i++) {
    onepass_ids_.push_back(val_idx[i].second);
  }
}

// Reset the internal variable to the initial state and reshuffle the indices.
Status WeightedRandomSamplerRT::ResetSampler() {
  sample_id_ = 0;
  buffer_id_ = 0;
  rand_gen_.seed(GetSeed());
  if (!replacement_) {
    InitOnePassSampling();
  } else {
    discrete_dist_->reset();
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

// Get the sample ids.
Status WeightedRandomSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (weights_.size() > static_cast<size_t>(num_rows_)) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid parameter, size of sample weights must be less than or equal to num of data, "
                  "otherwise might cause generated id out of bound or other errors, but got weight size: " +
                    std::to_string(weights_.size()) + ", num of data: " + std::to_string(num_rows_));
  }

  if (!replacement_ && (weights_.size() < static_cast<size_t>(num_samples_))) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, without replacement, weight size must be greater than or equal to num_samples, "
      "but got weight size: " +
      std::to_string(weights_.size()) + ", num_samples: " + std::to_string(num_samples_));
  }

  if (sample_id_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> outputIds;

    int64_t last_id = sample_id_ + samples_per_buffer_;
    // Handling the return all samples at once, and when last draw is not a full batch.
    if (last_id > num_samples_) {
      last_id = num_samples_;
    }

    // Allocate tensor.
    RETURN_IF_NOT_OK(CreateSamplerTensor(&outputIds, last_id - sample_id_));

    // Initialize tensor.
    auto id_ptr = outputIds->begin<int64_t>();
    // Assign the data to tensor element.
    while (sample_id_ < last_id) {
      int64_t genId;
      if (replacement_) {
        genId = (*discrete_dist_)(rand_gen_);
      } else {
        // Draw sample without replacement.
        genId = onepass_ids_.front();
        onepass_ids_.pop_front();
      }

      if (genId >= num_rows_) {
        RETURN_STATUS_UNEXPECTED("Generated indice is out of bound, expect range [0, num_data-1], got indice: " +
                                 std::to_string(genId) + ", num_data: " + std::to_string(num_rows_ - 1));
      }

      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&genId, genId));
      }

      *id_ptr = genId;
      id_ptr++;
      sample_id_++;
    }

    // Create a TensorTable from that single tensor and push into DataBuffer
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, TensorRow(1, outputIds)));
  }

  return Status::OK();
}

void WeightedRandomSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: WeightedRandomSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status WeightedRandomSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sampler_name"] = "WeightedRandomSampler";
  args["weights"] = weights_;
  args["num_samples"] = num_samples_;
  args["replacement"] = replacement_;
  if (this->HasChildSampler()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : child_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
