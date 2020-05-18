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
#include "dataset/engine/datasetops/source/sampler/subset_random_sampler.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "dataset/core/config_manager.h"
#include "dataset/core/global_context.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
// Constructor.
SubsetRandomSampler::SubsetRandomSampler(const std::vector<int64_t> &indices, int64_t samples_per_buffer)
    : Sampler(samples_per_buffer), indices_(indices), sample_id_(0), buffer_id_(0) {}

// Initialized this Sampler.
Status SubsetRandomSampler::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0, "num_rows <= 0\n");

  num_samples_ = indices_.size();

  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  if (static_cast<size_t>(samples_per_buffer_) > indices_.size()) {
    samples_per_buffer_ = static_cast<int64_t>(indices_.size());
  }

  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return Status::OK();
}

// Reset the internal variable to the initial state.
Status SubsetRandomSampler::Reset() {
  // Reset the internal counters.
  sample_id_ = 0;
  buffer_id_ = 0;

  // Randomized the indices again.
  rand_gen_.seed(GetSeed());
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

// Get the sample ids.
Status SubsetRandomSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  // All samples have been drawn
  if (sample_id_ == indices_.size()) {
    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> outputIds;

    int64_t last_id = sample_id_ + samples_per_buffer_;
    // Handling the return all samples at once, and when last draw is not a full batch.
    if (static_cast<size_t>(last_id) > indices_.size()) {
      last_id = indices_.size();
    }

    // Allocate tensor
    RETURN_IF_NOT_OK(CreateSamplerTensor(&outputIds, last_id - sample_id_));

    // Initialize tensor
    int64_t *id_ptr = reinterpret_cast<int64_t *>(outputIds->GetMutableBuffer());
    while (sample_id_ < last_id) {
      if (indices_[sample_id_] >= num_rows_) {
        std::string err_msg =
          "Generated id is bigger than numRows (out of bound). indices_: " + std::to_string(indices_[sample_id_]) +
          " num_rows_: " + std::to_string(num_rows_);
        RETURN_STATUS_UNEXPECTED(err_msg);
      }

      int64_t sampled_id = indices_[sample_id_];
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
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
