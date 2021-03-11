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
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace mindspore {
namespace dataset {
SequentialSamplerRT::SequentialSamplerRT(int64_t num_samples, int64_t start_index, int64_t samples_per_buffer)
    : SamplerRT(num_samples, samples_per_buffer), current_id_(start_index), start_index_(start_index), id_count_(0) {}

Status SequentialSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
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

Status SequentialSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ >= 0,
                               "Invalid parameter, start_index must be greater than or equal to 0, but got " +
                                 std::to_string(start_index_) + ".\n");
  CHECK_FAIL_RETURN_UNEXPECTED(start_index_ < num_rows_,
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
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_samples_ > 0 && samples_per_buffer_ > 0,
    "Invalid parameter, samples_per_buffer must be greater than 0, but got " + std::to_string(samples_per_buffer_));
  samples_per_buffer_ = samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;

  is_initialized = true;
  return Status::OK();
}

Status SequentialSamplerRT::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(id_count_ == num_samples_, "ERROR Reset() called early/late");
  current_id_ = start_index_;
  id_count_ = 0;

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
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
    if (child_num_rows == -1) return child_num_rows;
  }
  int64_t num_samples = (num_samples_ > 0) ? std::min(child_num_rows, num_samples_) : child_num_rows;
  // For this sampler we need to take start_index into account. Because for example in the case we are given n rows
  // and start_index != 0 and num_samples >= n then we can't return all the n rows.
  if (child_num_rows - start_index_ <= 0) {
    return 0;
  }
  if (child_num_rows - start_index_ < num_samples)
    num_samples = child_num_rows - start_index_ > num_samples ? num_samples : num_samples - start_index_;
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
  nlohmann::json args;
  args["sampler_name"] = "SequentialSampler";
  args["start_index"] = start_index_;
  args["num_samples"] = num_samples_;
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
