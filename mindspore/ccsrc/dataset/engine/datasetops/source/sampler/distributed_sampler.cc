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
#include "dataset/engine/datasetops/source/sampler/distributed_sampler.h"

#include <limits>
#include <memory>

#include "dataset/engine/data_buffer.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
DistributedSampler::DistributedSampler(int64_t num_samples, int64_t num_dev, int64_t dev_id, bool shuffle,
                                       uint32_t seed)
    : Sampler(num_samples, std::numeric_limits<int64_t>::max()),
      cnt_(0),
      seed_(seed == std::numeric_limits<uint32_t>::max() ? GetSeed() : seed),
      device_id_(dev_id),
      num_devices_(num_dev),
      shuffle_(shuffle) {}

Status DistributedSampler::InitSampler() {
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0, "num_samples <= 0\n");
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0, "num_rows <= 0\n");
  CHECK_FAIL_RETURN_UNEXPECTED(device_id_ < num_devices_ && device_id_ >= 0 && num_rows_ > 0 && num_samples_ > 0,
                               "fail to init DistributedSampler");
  rnd_.seed(seed_++);
  samples_per_buffer_ = (num_rows_ + num_devices_ - 1) / num_devices_;  // equals to ceil(num_rows/num_devices)
  samples_per_buffer_ = num_samples_ < samples_per_buffer_ ? num_samples_ : samples_per_buffer_;
  if (shuffle_ == true) {
    shuffle_vec_.reserve(num_rows_);
    for (int64_t i = 0; i < num_rows_; i++) {
      shuffle_vec_.push_back(i);
    }
    std::shuffle(shuffle_vec_.begin(), shuffle_vec_.end(), rnd_);
  }
  return Status::OK();
}

Status DistributedSampler::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (cnt_ > samples_per_buffer_) {
    RETURN_STATUS_UNEXPECTED("Distributed Sampler Error");
  } else if (cnt_ == samples_per_buffer_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(cnt_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sample_ids;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, samples_per_buffer_));
    auto id_ptr = sample_ids->begin<int64_t>();
    while (cnt_ < samples_per_buffer_ && id_ptr != sample_ids->end<int64_t>()) {
      int64_t sampled_id = (num_devices_ * cnt_ + device_id_) % num_rows_;
      if (shuffle_) {
        sampled_id = shuffle_vec_[static_cast<size_t>(sampled_id)];
      }

      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
      id_ptr++;
      cnt_++;
    }
    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status DistributedSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(cnt_ == samples_per_buffer_, "ERROR Reset() called early/late");
  cnt_ = 0;

  if (shuffle_ == true) {
    rnd_.seed(seed_);
    seed_++;
    std::shuffle(shuffle_vec_.begin(), shuffle_vec_.end(), rnd_);
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

void DistributedSampler::Print(std::ostream &out, bool show_all) const {
  out << "(sampler): DistributedSampler\n";
  if (show_all) {
    out << "seed_: " << seed_ << '\n';
    out << "device_id_: " << device_id_ << '\n';
    out << "num_devices_: " << num_devices_ << '\n';
    out << "shuffle_: " << shuffle_ << '\n';
  }
}

}  // namespace dataset
}  // namespace mindspore
