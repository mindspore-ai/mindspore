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
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
DistributedSamplerRT::DistributedSamplerRT(int64_t num_samples, int64_t num_dev, int64_t dev_id, bool shuffle,
                                           uint32_t seed, int64_t offset, bool even_dist)
    : SamplerRT(num_samples, std::numeric_limits<int64_t>::max()),
      cnt_(0),
      seed_(seed == std::numeric_limits<uint32_t>::max() ? GetSeed() : seed),
      device_id_(dev_id),
      num_devices_(num_dev),
      shuffle_(shuffle),
      even_dist_(even_dist),
      offset_(offset),
      non_empty_(true) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_devices_);
}

Status DistributedSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0, "Invalid parameter, num_samples must be greater than 0, but got " +
                                                   std::to_string(num_samples_) + ".\n");
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0, "Invalid parameter, num_rows must be greater than 0" + std::to_string(num_rows_) + ".\n");
  CHECK_FAIL_RETURN_UNEXPECTED(
    device_id_ < num_devices_ && device_id_ >= 0 && num_rows_ > 0 && num_samples_ > 0,
    "Invalid parameter, num_shard must be greater than shard_id and greater than 0, got num_shard: " +
      std::to_string(num_devices_) + ", shard_id: " + std::to_string(device_id_) + ".\n");
  rnd_.seed(seed_++);

  if (offset_ != -1 || !even_dist_) {
    if (offset_ == -1) offset_ = 0;
    samples_per_buffer_ = (num_rows_ + offset_) / num_devices_;
    int64_t remainder = (num_rows_ + offset_) % num_devices_;
    if (device_id_ < remainder) samples_per_buffer_++;
    if (device_id_ < offset_) samples_per_buffer_--;
  } else {
    offset_ = 0;
    samples_per_buffer_ = (num_rows_ + num_devices_ - 1) / num_devices_;  // equals to ceil(num_rows/num_devices)
  }
  samples_per_buffer_ = num_samples_ < samples_per_buffer_ ? num_samples_ : samples_per_buffer_;
  if (shuffle_) {
    shuffle_vec_.reserve(num_rows_);
    for (int64_t i = 0; i < num_rows_; i++) {
      shuffle_vec_.push_back(i);
    }
    std::shuffle(shuffle_vec_.begin(), shuffle_vec_.end(), rnd_);
  }
  if (!samples_per_buffer_) non_empty_ = false;

  is_initialized = true;
  return Status::OK();
}

Status DistributedSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (cnt_ > samples_per_buffer_) {
    RETURN_STATUS_UNEXPECTED(
      "Number of samples(cnt) that have already been filled in to buffer should be less than or "
      "equal to samples_per_buffer, but got cnt: " +
      std::to_string(cnt_) + ", samples_per_buffer: " + std::to_string(samples_per_buffer_));
  } else if (cnt_ == samples_per_buffer_ && (non_empty_ || !even_dist_)) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    if (!samples_per_buffer_) {
      non_empty_ = false;
    }
  } else if (!samples_per_buffer_ && !non_empty_) {
    // If the buffer is empty, we add samples with subscript 0 in the current dataset.
    // This step is to make up for the solution that the code default buffer is not empty before.
    // We will remove this value in the concat phase
    non_empty_ = true;
    (*out_buffer) = std::make_unique<DataBuffer>(cnt_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sample_ids;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, 1));
    auto id_ptr = sample_ids->begin<int64_t>();
    // add index 0
    *id_ptr = 0;
    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(cnt_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sample_ids;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, samples_per_buffer_));
    auto id_ptr = sample_ids->begin<int64_t>();
    bool flag_add_1 = false;
    while (cnt_ < samples_per_buffer_ && id_ptr != sample_ids->end<int64_t>()) {
      int64_t middle_value = num_devices_ * cnt_ + device_id_ - offset_;
      // if index < 0, we move back one place
      if (middle_value < 0) {
        samples_per_buffer_++;
        cnt_++;
        flag_add_1 = true;
        middle_value = num_devices_ * cnt_ + device_id_ - offset_;
      }
      int64_t sampled_id = middle_value % num_rows_;

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

    // If 1 was added before, we will cut off 1 here
    if (flag_add_1) {
      samples_per_buffer_--;
      cnt_--;
    }
    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status DistributedSamplerRT::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(cnt_ == samples_per_buffer_, "ERROR Reset() called early/late");
  cnt_ = 0;

  if (shuffle_ == true) {
    rnd_.seed(seed_);
    seed_++;
    std::shuffle(shuffle_vec_.begin(), shuffle_vec_.end(), rnd_);
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

int64_t DistributedSamplerRT::CalculateNumSamples(int64_t num_rows) {
  int64_t child_num_rows = num_rows;
  if (!child_.empty()) {
    child_num_rows = child_[0]->CalculateNumSamples(num_rows);
  }
  int64_t num_samples = (num_samples_ > 0) ? std::min(child_num_rows, num_samples_) : child_num_rows;
  int64_t remainder = (child_num_rows + offset_) % num_devices_;
  int64_t shard_size = (child_num_rows + offset_) / num_devices_;
  if (offset_ != -1 || !even_dist_) {
    if (offset_ == -1) offset_ = 0;
    if (device_id_ < remainder) shard_size++;
    if (device_id_ < offset_) shard_size--;
  } else {
    shard_size = (child_num_rows + num_devices_ - 1) / num_devices_;
  }
  // add 1 to an empty shard
  // this logic is needed to follow the logic in initSampler that is written for ConcatDataset
  if (shard_size == 0) shard_size++;

  return std::min(num_samples, shard_size);
}

void DistributedSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: DistributedSampler";
  if (show_all) {
    SamplerRT::SamplerPrint(out, show_all);
    out << "\nseed: " << seed_ << "\ndevice_id: " << device_id_ << "\nnum_devices: " << num_devices_
        << "\nshuffle: " << shuffle_;
  }
}

Status DistributedSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sampler_name"] = "DistributedSampler";
  args["num_shards"] = num_devices_;
  args["shard_id"] = device_id_;
  args["shuffle"] = shuffle_;
  args["num_samples"] = num_samples_;
  args["offset"] = offset_;
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
