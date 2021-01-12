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
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"

#include <algorithm>
#include <limits>
#include <memory>
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
RandomSamplerRT::RandomSamplerRT(int64_t num_samples, bool replacement, bool reshuffle_each_epoch,
                                 int64_t samples_per_buffer)
    : SamplerRT(num_samples, samples_per_buffer),
      seed_(GetSeed()),
      replacement_(replacement),
      next_id_(0),
      dist(nullptr),
      reshuffle_each_epoch_(reshuffle_each_epoch) {}

Status RandomSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (next_id_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED("RandomSampler Internal Error");
  } else if (next_id_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }
    (*out_buffer) = std::make_unique<DataBuffer>(next_id_, DataBuffer::kDeBFlagNone);

    std::shared_ptr<Tensor> sampleIds;
    int64_t last_id = std::min(samples_per_buffer_ + next_id_, num_samples_);
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, last_id - next_id_));
    auto id_ptr = sampleIds->begin<int64_t>();

    for (int64_t i = 0; i < (last_id - next_id_); i++) {
      int64_t sampled_id = 0;
      if (replacement_) {
        sampled_id = (*dist)(rnd_);
      } else {
        sampled_id = shuffled_ids_[static_cast<size_t>(i + next_id_)];
      }

      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *(id_ptr + static_cast<ptrdiff_t>(i)) = sampled_id;
    }
    next_id_ = last_id;
    TensorRow row(1, sampleIds);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status RandomSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_samples_ > 0 && num_rows_ > 0,
    "Invalid parameter, num_samples & num_rows must be greater than 0, but got num_samples: " +
      std::to_string(num_samples_) + ", num_rows: " + std::to_string(num_rows_));
  samples_per_buffer_ = samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;
  rnd_.seed(seed_);

  if (!replacement_) {
    shuffled_ids_.reserve(num_rows_);
    for (int64_t i = 0; i < num_rows_; i++) {
      shuffled_ids_.push_back(i);
    }
    std::shuffle(shuffled_ids_.begin(), shuffled_ids_.end(), rnd_);
  } else {
    dist = std::make_unique<std::uniform_int_distribution<int64_t>>(0, num_rows_ - 1);
  }

  is_initialized = true;
  return Status::OK();
}

Status RandomSamplerRT::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(next_id_ == num_samples_, "ERROR Reset() called early/late");
  next_id_ = 0;

  if (reshuffle_each_epoch_) {
    seed_++;
  }

  rnd_.seed(seed_);

  if (!replacement_ && reshuffle_each_epoch_) {
    std::shuffle(shuffled_ids_.begin(), shuffled_ids_.end(), rnd_);
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

void RandomSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: RandomSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status RandomSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sampler_name"] = "RandomSampler";
  args["replacement"] = replacement_;
  args["num_samples"] = num_samples_;
  args["reshuffle_each_epoch"] = reshuffle_each_epoch_;
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
