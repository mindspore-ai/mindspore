/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/sampler/subset_sampler.h"

namespace mindspore {
namespace dataset {
// Constructor.
SubsetSamplerRT::SubsetSamplerRT(const std::vector<int64_t> &indices, int64_t num_samples, int64_t samples_per_tensor)
    : SamplerRT(num_samples, samples_per_tensor), indices_(indices), sample_id_(0) {}

// Initialized this Sampler.
Status SubsetSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0, "Invalid parameter, num_rows must be greater than 0, but got " + std::to_string(num_rows_) + ".\n");

  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // In this case, the id's are provided by the user.  Cap the num_samples on the number of id's given.
  if (num_samples_ == 0 || num_samples_ > static_cast<int64_t>(indices_.size())) {
    num_samples_ = static_cast<int64_t>(indices_.size());
  }

  if (samples_per_tensor_ > num_samples_) {
    samples_per_tensor_ = num_samples_;
  }

  is_initialized = true;
  return Status::OK();
}

// Reset the internal variable to the initial state.
Status SubsetSamplerRT::ResetSampler() {
  // Reset the internal counters.
  sample_id_ = 0;

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

// Get the sample ids.
Status SubsetSamplerRT::GetNextSample(TensorRow *out) {
  // All samples have been drawn
  if (sample_id_ == num_samples_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> outputIds;

    int64_t last_id = sample_id_ + samples_per_tensor_;
    // Handling the return all samples at once, and when last draw is not a full batch.
    if (last_id > num_samples_) {
      last_id = num_samples_;
    }

    // Allocate tensor
    RETURN_IF_NOT_OK(CreateSamplerTensor(&outputIds, last_id - sample_id_));

    // Initialize tensor
    auto id_ptr = outputIds->begin<int64_t>();
    while (sample_id_ < last_id) {
      if (indices_[sample_id_] >= num_rows_ || indices_[sample_id_] < 0) {
        std::string err_msg = "Sample ID (" + std::to_string(indices_[sample_id_]) +
                              ") is out of bound, expected range [0, " + std::to_string(num_rows_ - 1) + "]";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }

      int64_t sampled_id = ((indices_[sample_id_] % num_rows_) + num_rows_) % num_rows_;
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
      ++id_ptr;
      sample_id_++;
    }

    (*out) = {outputIds};
  }

  return Status::OK();
}

void SubsetSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: SubsetSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status SubsetSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerRT::to_json(&args));
  args["sampler_name"] = "SubsetSampler";
  args["indices"] = indices_;

  *out_json = args;
  return Status::OK();
}

int64_t SubsetSamplerRT::CalculateNumSamples(int64_t num_rows) {
  int64_t child_num_rows = num_rows;
  if (!child_.empty()) {
    child_num_rows = child_[0]->CalculateNumSamples(num_rows);
    // return -1 if child_num_rows is undetermined
    if (child_num_rows == -1) return child_num_rows;
  }
  int64_t res = (num_samples_ > 0) ? std::min(child_num_rows, num_samples_) : child_num_rows;
  res = std::min(res, static_cast<int64_t>(indices_.size()));
  return res;
}
}  // namespace dataset
}  // namespace mindspore
