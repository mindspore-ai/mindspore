/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/sampler/skip_first_epoch_sampler.h"

#include <algorithm>
#include <memory>
#include <string>

namespace mindspore {
namespace dataset {
SkipFirstEpochSamplerRT::SkipFirstEpochSamplerRT(int64_t start_index, int64_t num_samples, int64_t samples_per_tensor)
    : SequentialSamplerRT(start_index, num_samples, samples_per_tensor), sample_need_to_skip_(start_index) {}

Status SkipFirstEpochSamplerRT::GetNextSample(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  if (index_produced_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED(
      "[Internal ERROR] Sampler index must be less than or equal to num_samples(total rows in dataset), but got:" +
      std::to_string(index_produced_) + ", num_samples_: " + std::to_string(num_samples_));
  } else if (index_produced_ == num_samples_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    int64_t num_index_valid;
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
      CHECK_FAIL_RETURN_UNEXPECTED(!child_ids_.empty(),
                                   "SkipFirstEpochSampler: got empty output index from child sampler.");
      num_index_valid = child_ids_[0]->Size();
      while (sample_need_to_skip_ >= num_index_valid) {
        sample_need_to_skip_ -= num_index_valid;
        current_index_ += num_index_valid;
        RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
        CHECK_FAIL_RETURN_UNEXPECTED(!child_ids_.empty(),
                                     "SkipFirstEpochSampler: got empty output index from child sampler.");
        num_index_valid = child_ids_[0]->Size();
      }
      CHECK_FAIL_RETURN_UNEXPECTED(
        child_ids_[0]->type().value() == DataType::Type::DE_INT64,
        "SkipFirstEpochSampler: output index from child sampler must be of int64 type, but got: " +
          child_ids_[0]->type().ToString());
    } else {
      // calculate how many ids left to produce
      num_index_valid = num_rows_ - index_produced_;
    }

    // Compute how many ids are left to pack, and pack this amount into a new Tensor.
    // Respect the setting for samples per Tensor though.
    int64_t remaining_ids = num_index_valid - sample_need_to_skip_;
    int64_t num_elements = std::min(remaining_ids, samples_per_tensor_);

    std::shared_ptr<Tensor> sampleIds;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, num_elements));

    if (HasChildSampler()) {
      std::string err_msg = "Failed to copy full sample ids into child sampler.";
      int64_t copy_data_length = num_elements * sizeof(int64_t);
      if (copy_data_length < SECUREC_MEM_MAX_LEN) {
        int ret_code =
          memcpy_s(sampleIds->GetMutableBuffer(), copy_data_length,
                   child_ids_[0]->GetMutableBuffer() + sample_need_to_skip_ * sizeof(int64_t), copy_data_length);
        CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK, err_msg + " errno: " + std::to_string(ret_code));
      } else {
        auto dest =
          std::memcpy(sampleIds->GetMutableBuffer(),
                      child_ids_[0]->GetMutableBuffer() + sample_need_to_skip_ * sizeof(int64_t), copy_data_length);
        CHECK_FAIL_RETURN_UNEXPECTED(dest == sampleIds->GetMutableBuffer(), err_msg);
      }
      current_index_ += num_elements;
    } else {
      auto idPtr = sampleIds->begin<int64_t>();
      for (int64_t i = 0; i < num_elements; i++) {
        *idPtr = current_index_;
        current_index_++;  // Move the current id to the next one in the sequence
        ++idPtr;
      }
    }
    index_produced_ += num_elements;  // Count the packed ids towards our overall sample count
    sample_need_to_skip_ = 0;
    (*out) = {sampleIds};
  }
  return Status::OK();
}

Status SkipFirstEpochSamplerRT::ResetSampler(const bool failover_reset) {
  // This is a special sampler for Failover Reset, its internal state should
  // not reset when failover_reset is set to true.
  if (!failover_reset) {
    if (index_produced_ != num_samples_) {
      std::string err_msg =
        "[Internal ERROR] ResetSampler() called early or late. index_produced_: " + std::to_string(index_produced_) +
        " num_samples_: " + std::to_string(num_samples_);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    current_index_ = 0;
    index_produced_ = 0;

    if (!first_epoch_done_) {
      num_samples_ += start_index_;
      start_index_ = 0;
      samples_per_tensor_ = num_samples_;
      first_epoch_done_ = true;
    }
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler(failover_reset));
  }

  return Status::OK();
}

int64_t SkipFirstEpochSamplerRT::CalculateNumSamples(const int64_t num_rows) { return -1; }

void SkipFirstEpochSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: SkipFirstEpochSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info
    out << "\nStart index: " << start_index_;
    out << "\nFirst epoch done: " << first_epoch_done_;
    out << "\nCurrent index: " << current_index_;
    out << "\nIndex produced:" << index_produced_;
  }
}

Status SkipFirstEpochSamplerRT::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerRT::to_json(&args));
  args["sampler_name"] = "SkipFirstEpochSampler";
  args["start_index"] = start_index_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
