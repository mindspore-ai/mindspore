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
DistributedSampler::DistributedSampler(int64_t num_dev, int64_t dev_id, bool shuffle, uint32_t seed)
    : Sampler(),
      cnt_(0),
      seed_(seed == std::numeric_limits<uint32_t>::max() ? GetSeed() : seed),
      device_id_(dev_id),
      num_devices_(num_dev),
      shuffle_(shuffle) {}

Status DistributedSampler::Init(const RandomAccessOp *op) {
  RETURN_IF_NOT_OK(Sampler::Init(op));
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

Status DistributedSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  if (cnt_ > samples_per_buffer_) {
    RETURN_STATUS_UNEXPECTED("Distributed Sampler Error");
  } else if (cnt_ == samples_per_buffer_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    (*out_buffer) = std::make_unique<DataBuffer>(cnt_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sample_ids;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, samples_per_buffer_));
    int64_t *id_ptr = reinterpret_cast<int64_t *>(sample_ids->StartAddr());
    while (cnt_ < samples_per_buffer_) {
      int64_t next_id = (num_devices_ * (cnt_++) + device_id_) % num_rows_;
      *(id_ptr++) = shuffle_ ? shuffle_vec_[static_cast<size_t>(next_id)] : next_id;
    }
    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status DistributedSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(cnt_ == samples_per_buffer_, "ERROR Reset() called early/late");
  cnt_ = 0;
  rnd_.seed(seed_++);
  if (shuffle_ == true) {
    std::shuffle(shuffle_vec_.begin(), shuffle_vec_.end(), rnd_);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
