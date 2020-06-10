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
#include "dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include <algorithm>
#include <memory>
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
PKSampler::PKSampler(int64_t num_samples, int64_t val, bool shuffle, int64_t samples_per_buffer)
    : Sampler(num_samples, samples_per_buffer),
      shuffle_(shuffle),
      seed_(GetSeed()),
      next_id_(0),
      samples_per_class_(val) {}

Status PKSampler::InitSampler() {
  labels_.reserve(label_to_ids_.size());
  for (const auto &pair : label_to_ids_) {
    if (pair.second.empty() == false) {
      labels_.push_back(pair.first);
    }
  }
  rnd_.seed(seed_++);

  // The special handshake gives the list of classes and id's, but it did not set the num_rows_ to
  // capture the total number of possible sample ids.
  // Compute that here for this case to find the total number of samples that are available to return.
  // (in this case, samples per class * total classes).
  num_rows_ = samples_per_class_ * static_cast<int64_t>(labels_.size());

  // The user may have chosen to sample less than the total amount.
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }

  samples_per_buffer_ = (samples_per_buffer_ > num_samples_) ? num_samples_ : samples_per_buffer_;
  if (shuffle_ == true) {
    std::shuffle(labels_.begin(), labels_.end(), rnd_);
  } else {
    std::sort(labels_.begin(), labels_.end());
  }
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0, "num_class or K (num samples per class) is not positive");
  return Status::OK();
}

Status PKSampler::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (next_id_ > num_samples_ || num_samples_ == 0) {
    RETURN_STATUS_UNEXPECTED("Index out of bound in PKSampler");
  } else if (next_id_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(next_id_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sample_ids;
    int64_t last_id = (samples_per_buffer_ + next_id_ > num_samples_) ? num_samples_ : samples_per_buffer_ + next_id_;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, last_id - next_id_));
    int64_t *id_ptr = reinterpret_cast<int64_t *>(sample_ids->GetMutableBuffer());
    while (next_id_ < last_id) {
      int64_t cls_id = next_id_++ / samples_per_class_;
      const std::vector<int64_t> &samples = label_to_ids_[labels_[cls_id]];
      int64_t rnd_ind = std::uniform_int_distribution<int64_t>(0, samples.size() - 1)(rnd_);
      int64_t sampled_id = samples[rnd_ind];

      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
      id_ptr++;
    }

    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status PKSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(next_id_ == num_samples_, "ERROR Reset() called early/late");
  next_id_ = 0;
  rnd_.seed(seed_++);

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

Status PKSampler::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  RETURN_UNEXPECTED_IF_NULL(op);
  RETURN_IF_NOT_OK(op->GetClassIds(&label_to_ids_));
  RETURN_IF_NOT_OK(InitSampler());
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
