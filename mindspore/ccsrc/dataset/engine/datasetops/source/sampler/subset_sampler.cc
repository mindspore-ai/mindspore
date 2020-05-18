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
#include "dataset/engine/datasetops/source/sampler/subset_sampler.h"

#include <memory>
#include <string>

#include "dataset/core/config_manager.h"
#include "dataset/core/global_context.h"

namespace mindspore {
namespace dataset {
// Constructor.
SubsetSampler::SubsetSampler(int64_t start_index, int64_t subset_size)
    : Sampler(subset_size), start_index_(start_index), subset_size_(subset_size), current_id_(0) {}

Status SubsetSampler::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(subset_size_ > 0, "subset_size_ <= 0\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ >= 0, "start_index < 0\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ < num_rows_, "start_index >= num_rows_\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ + subset_size_ - 1 < num_rows_, "Final index out of bounds.\n");

  num_samples_ = subset_size_;

  return Status::OK();
}

Status SubsetSampler::Reset() {
  current_id_ = 0;

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->Reset());
  }

  return Status::OK();
}

Status SubsetSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  if (current_id_ > subset_size_) {
    RETURN_STATUS_UNEXPECTED("SubsetSampler Internal Error");
  } else if (current_id_ == subset_size_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sampled_ids;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampled_ids, subset_size_));

    int64_t *sampled_ids_start_addr = reinterpret_cast<int64_t *>(sampled_ids->GetMutableBuffer());

    while (current_id_ < subset_size_) {
      int64_t sampled_id = start_index_ + current_id_;
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *(sampled_ids_start_addr + current_id_) = sampled_id;
      current_id_++;
    }

    TensorRow sampled_ids_row(1, sampled_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, sampled_ids_row));
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
