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
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include <limits>
#include <memory>
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
RandomSampler::RandomSampler(bool replacement, int64_t num_samples, int64_t samples_per_buffer)
    : Sampler(samples_per_buffer),
      seed_(GetSeed()),
      replacement_(replacement),
      user_num_samples_(num_samples),
      next_id_(0),
      dist(nullptr) {}

Status RandomSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  if (next_id_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED("RandomSampler Internal Error");
  } else if (next_id_ == num_samples_) {
    (*out_buffer) = make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    (*out_buffer) = make_unique<DataBuffer>(next_id_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sampleIds;
    int64_t last_id = samples_per_buffer_ + next_id_ > num_samples_ ? num_samples_ : samples_per_buffer_ + next_id_;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, last_id - next_id_));
    int64_t *id_ptr = reinterpret_cast<int64_t *>(sampleIds->StartAddr());
    for (int64_t i = 0; i < (last_id - next_id_); i++) {
      *(id_ptr + i) = replacement_ ? (*dist)(rnd_) : shuffled_ids_[static_cast<size_t>(i + next_id_)];
    }
    next_id_ = last_id;
    TensorRow row(1, sampleIds);
    (*out_buffer)->set_tensor_table(make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status RandomSampler::Init(const RandomAccessOp *op) {
  RETURN_IF_NOT_OK(Sampler::Init(op));
  num_samples_ = (user_num_samples_ < num_samples_) ? user_num_samples_ : num_samples_;
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0 && num_rows_ > 0, "Fail to init RandomSampler");
  samples_per_buffer_ = samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;
  if (replacement_ == false) {
    shuffled_ids_.reserve(num_rows_);
    for (int64_t i = 0; i < num_rows_; i++) {
      shuffled_ids_.push_back(i);
    }
    std::shuffle(shuffled_ids_.begin(), shuffled_ids_.end(), rnd_);
  } else {
    dist = make_unique<std::uniform_int_distribution<int64_t>>(0, num_rows_ - 1);
  }
  rnd_.seed(seed_++);
  return Status::OK();
}

Status RandomSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(next_id_ == num_samples_, "ERROR Reset() called early/late");
  next_id_ = 0;
  rnd_.seed(seed_++);
  if (replacement_ == false) {
    std::shuffle(shuffled_ids_.begin(), shuffled_ids_.end(), rnd_);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
