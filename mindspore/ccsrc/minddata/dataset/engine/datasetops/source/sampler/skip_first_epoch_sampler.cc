/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <string>

namespace mindspore {
namespace dataset {
Status SkipFirstEpochSamplerRT::ResetSampler(const bool failover_reset) {
  // This is a special sampler for Failover Reset, its internal state should
  // not reset when failover_reset is set to true.
  if (!failover_reset) {
    if (id_count_ != num_samples_) {
      std::string err_msg =
        "[Internal ERROR] ResetSampler() called early or late. id_count_: " + std::to_string(id_count_) +
        " num_samples_: " + std::to_string(num_samples_);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    current_id_ = 0;
    id_count_ = 0;

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
    out << "\nCurrent id: " << current_id_;
    out << "\nid count:" << id_count_;
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
