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
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"

#include <algorithm>
#include <memory>

namespace mindspore {
namespace dataset {
SequentialSampler::SequentialSampler(int64_t num_samples, int64_t start_index, int64_t samples_per_buffer)
    : Sampler(num_samples, samples_per_buffer), start_index_(start_index), current_id_(start_index), id_count_(0) {}

Status SequentialSampler::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (id_count_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED("SequentialSampler Internal Error");
  } else if (id_count_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(current_id_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sampleIds;

    // Compute how many ids are left to pack, and pack this amount into a new buffer.  Respect the setting for
    // samples per buffer though.
    int64_t remaining_ids = num_samples_ - id_count_;
    int64_t num_elements = std::min(remaining_ids, samples_per_buffer_);

    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, num_elements));
    auto idPtr = sampleIds->begin<int64_t>();
    for (int64_t i = 0; i < num_elements; i++) {
      int64_t sampled_id = current_id_;
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *idPtr = sampled_id;
      current_id_++;  // Move the current id to the next one in the sequence
      idPtr++;
    }

    id_count_ += num_elements;  // Count the packed ids towards our overall sample count

    TensorRow row(1, sampleIds);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status SequentialSampler::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ >= 0, "start_index < 0\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ < num_rows_, "start_index >= num_rows\n");
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ >= 0, "num_samples < 0\n");
  // Adjust the num_samples count based on the range of ids we are sequencing.  If num_samples is 0, we sample
  // the entire set.  If it's non-zero, we will implicitly cap the amount sampled based on available data.
  int64_t available_row_count = num_rows_ - start_index_;
  if (num_samples_ == 0 || num_samples_ > available_row_count) {
    num_samples_ = available_row_count;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0 && samples_per_buffer_ > 0, "Fail to init Sequential Sampler");
  samples_per_buffer_ = samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;
  return Status::OK();
}

Status SequentialSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(id_count_ == num_samples_, "ERROR Reset() called early/late");
  current_id_ = start_index_;
  id_count_ = 0;

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

void SequentialSampler::Print(std::ostream &out, bool show_all) const { out << "(sampler): SequentialSampler\n"; }

}  // namespace dataset
}  // namespace mindspore
