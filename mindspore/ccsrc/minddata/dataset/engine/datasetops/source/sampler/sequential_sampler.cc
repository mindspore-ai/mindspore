/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace mindspore {
namespace dataset {
SequentialSamplerRT::SequentialSamplerRT(int64_t start_index, int64_t num_samples, int64_t samples_per_tensor)
    : SamplerRT(num_samples, samples_per_tensor),
      current_index_(start_index),
      start_index_(start_index),
      index_produced_(0) {}

Status SequentialSamplerRT::GetNextSample(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  if (index_produced_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED(
      "[Internal ERROR] Sampler index must be less than or equal to num_samples(total rows in dataset), but got:" +
      std::to_string(index_produced_) + ", num_samples_: " + std::to_string(num_samples_));
  } else if (index_produced_ == num_samples_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> sampleIds;

    // Compute how many ids are left to pack, and pack this amount into a new Tensor.  Respect the setting for
    // samples per Tensor though.
    int64_t remaining_ids = num_samples_ - index_produced_;
    int64_t num_elements = std::min(remaining_ids, samples_per_tensor_);

    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, num_elements));
    auto idPtr = sampleIds->begin<int64_t>();
    for (int64_t i = 0; i < num_elements; i++) {
      int64_t sampled_id = current_index_;
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *idPtr = sampled_id;
      current_index_++;  // Move the current id to the next one in the sequence
      ++idPtr;
    }

    index_produced_ += num_elements;  // Count the packed ids towards our overall sample count

    (*out) = {sampleIds};
  }
  return Status::OK();
}

Status SequentialSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ >= 0,
                               "Invalid parameter, start_index must be greater than or equal to 0, but got " +
                                 std::to_string(start_index_) + ".\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ < num_rows_ || (num_rows_ == 0 && start_index_ == 0),
                               "Invalid parameter, start_index must be less than num_rows, but got start_index: " +
                                 std::to_string(start_index_) + ", num_rows: " + std::to_string(num_rows_) + ".\n");
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ >= 0,
                               "Invalid parameter, num_samples must be greater than or equal to 0, but got " +
                                 std::to_string(num_samples_) + ".\n");
  // Adjust the num_samples count based on the range of ids we are sequencing.  If num_samples is 0, we sample
  // the entire set.  If it's non-zero, we will implicitly cap the amount sampled based on available data.
  int64_t available_row_count = num_rows_ - start_index_;
  if (num_samples_ == 0 || num_samples_ > available_row_count) {
    num_samples_ = available_row_count;
  }
  CHECK_FAIL_RETURN_UNEXPECTED((num_samples_ > 0 && samples_per_tensor_ > 0) || num_samples_ == 0,
                               "Invalid parameter, samples_per_tensor(num_samplers) must be greater than 0, but got " +
                                 std::to_string(samples_per_tensor_));
  samples_per_tensor_ = samples_per_tensor_ > num_samples_ ? num_samples_ : samples_per_tensor_;

  is_initialized = true;
  return Status::OK();
}

Status SequentialSamplerRT::ResetSampler(const bool failover_reset) {
  CHECK_FAIL_RETURN_UNEXPECTED(failover_reset || index_produced_ == num_samples_,
                               "[Internal ERROR] ResetSampler() called early or late.");
  current_index_ = start_index_;
  index_produced_ = 0;

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler(failover_reset));
  }

  return Status::OK();
}

int64_t SequentialSamplerRT::CalculateNumSamples(int64_t num_rows) {
  // Holds the number of rows available for Sequential sampler. It can be the rows passed from its child sampler or the
  // num_rows from the dataset
  int64_t child_num_rows = num_rows;
  if (!child_.empty()) {
    child_num_rows = child_[0]->CalculateNumSamples(num_rows);
    // return -1 if child_num_rows is undetermined
    if (child_num_rows == -1) {
      return child_num_rows;
    }
  }
  int64_t num_samples = (num_samples_ > 0) ? std::min(child_num_rows, num_samples_) : child_num_rows;
  // For this sampler we need to take start_index into account. Because for example in the case we are given n rows
  // and start_index != 0 and num_samples >= n then we can't return all the n rows.
  if (child_num_rows - start_index_ <= 0) {
    return 0;
  }
  if (child_num_rows - start_index_ < num_samples) {
    num_samples = child_num_rows - start_index_ > num_samples ? num_samples : num_samples - start_index_;
  }
  return num_samples;
}

void SequentialSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: SequentialSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info
    out << "\nStart index: " << start_index_;
  }
}

Status SequentialSamplerRT::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerRT::to_json(&args));
  args["sampler_name"] = "SequentialSampler";
  args["start_index"] = start_index_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
