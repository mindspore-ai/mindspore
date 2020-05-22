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
#include "dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"

#include <algorithm>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "dataset/core/global_context.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
//  Constructor.
WeightedRandomSampler::WeightedRandomSampler(const std::vector<double> &weights, int64_t num_samples, bool replacement,
                                             int64_t samples_per_buffer)
    : Sampler(samples_per_buffer),
      weights_(weights),
      replacement_(replacement),
      sample_id_(0),
      buffer_id_(0),
      user_num_samples_(num_samples) {}

// Initialized this Sampler.
Status WeightedRandomSampler::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0 && user_num_samples_, "num_samples & num_rows need to be positive");
  CHECK_FAIL_RETURN_UNEXPECTED(samples_per_buffer_ > 0, "samples_per_buffer<=0\n");
  num_samples_ = user_num_samples_;

  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  samples_per_buffer_ = (samples_per_buffer_ > user_num_samples_) ? user_num_samples_ : samples_per_buffer_;

  if (!replacement_) {
    exp_dist_ = std::make_unique<std::exponential_distribution<>>(1);
    InitOnePassSampling();
  } else {
    discrete_dist_ = std::make_unique<std::discrete_distribution<int64_t>>(weights_.begin(), weights_.end());
  }

  return Status::OK();
}

// Initialized the computation for generating weighted random numbers without replacement using onepass method.
void WeightedRandomSampler::InitOnePassSampling() {
  exp_dist_->reset();
  onepass_ids_.clear();
  std::vector<std::pair<double, int64_t>> val_idx;
  for (size_t i = 0; i < weights_.size(); i++) {
    val_idx.emplace_back(std::make_pair((*exp_dist_)(rand_gen_) / weights_[i], i));
  }

  // Partial sort the first `numSamples` elements.
  std::partial_sort(val_idx.begin(), val_idx.begin() + user_num_samples_, val_idx.end());
  for (int64_t i = 0; i < user_num_samples_; i++) {
    onepass_ids_.push_back(val_idx[i].second);
  }
}

// Reset the internal variable to the initial state and reshuffle the indices.
Status WeightedRandomSampler::Reset() {
  sample_id_ = 0;
  buffer_id_ = 0;
  rand_gen_.seed(GetSeed());
  if (!replacement_) {
    InitOnePassSampling();
  } else {
    discrete_dist_->reset();
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

// Get the sample ids.
Status WeightedRandomSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  if (weights_.size() > static_cast<size_t>(num_rows_)) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "number of samples weights is more than num of rows. Might generate id out of bound OR other errors");
  }

  if (!replacement_ && (weights_.size() < static_cast<size_t>(user_num_samples_))) {
    RETURN_STATUS_UNEXPECTED("Without replacement, sample weights less than numSamples");
  }

  if (sample_id_ == user_num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> outputIds;

    int64_t last_id = sample_id_ + samples_per_buffer_;
    // Handling the return all samples at once, and when last draw is not a full batch.
    if (last_id > user_num_samples_) {
      last_id = user_num_samples_;
    }

    // Allocate tensor.
    RETURN_IF_NOT_OK(CreateSamplerTensor(&outputIds, last_id - sample_id_));

    // Initialize tensor.
    int64_t *id_ptr = reinterpret_cast<int64_t *>(outputIds->GetMutableBuffer());
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
        RETURN_STATUS_UNEXPECTED("generated id is bigger than numRows (out of bound).");
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
}  // namespace dataset
}  // namespace mindspore
