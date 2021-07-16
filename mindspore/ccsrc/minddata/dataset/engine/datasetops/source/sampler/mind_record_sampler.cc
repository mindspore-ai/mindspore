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
#include "minddata/dataset/engine/datasetops/source/sampler/mind_record_sampler.h"

#include "minddata/mindrecord/include/shard_reader.h"

namespace mindspore {
namespace dataset {
MindRecordSamplerRT::MindRecordSamplerRT(mindrecord::ShardReader *shard_reader, int64_t samples_per_tensor)
    : SamplerRT(0, samples_per_tensor), shard_reader_(shard_reader), sample_ids_(nullptr), next_id_(0) {}

Status MindRecordSamplerRT::GetNextSample(TensorRow *out) {
  if (next_id_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED(
      "Sampler index must be less than or equal to num_samples(total rows in dataset), but got: " +
      std::to_string(next_id_) + ", num_samples_: " + std::to_string(num_samples_));
  } else if (next_id_ == num_samples_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    std::shared_ptr<Tensor> sampleIdsTensor;
    int64_t last_id = std::min(samples_per_tensor_ + next_id_, num_samples_);
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIdsTensor, last_id - next_id_));
    auto id_ptr = sampleIdsTensor->begin<int64_t>();
    for (int64_t i = 0; i < (last_id - next_id_); i++) {
      *(id_ptr + static_cast<ptrdiff_t>(i)) = (*sample_ids_)[i];
    }
    next_id_ = last_id;

    (*out) = {sampleIdsTensor};
  }
  return Status::OK();
}

Status MindRecordSamplerRT::InitSampler() {
  sample_ids_ = shard_reader_->GetSampleIds();
  if (!sample_ids_) {
    // Note, sample_ids_.empty() is okay and will just give no sample ids.
    RETURN_STATUS_UNEXPECTED(
      "Init Sampler failed as sample_ids is empty, here ShardReader did not provide a valid sample ids vector via"
      " MindRecordSamplerRT");
  }

  // Usually, the num samples is given from the user interface. In our case, that data is in mindrecord.
  // Mindrecord already created the sample ids at this point, so the num samples is the size of the sampled id list.
  num_samples_ = sample_ids_->size();
  return Status::OK();
}

Status MindRecordSamplerRT::ResetSampler() {
  // drive the shard reader reshuffle tasks to redo the sampling for another epoch
  // Note that when cache is attached, this function is driven by cache lookup op rather than mindrecord op.
  // Therefore, the reshuffle of tasks might happen in the middle of mindrecord's epoch
  next_id_ = 0;
  shard_reader_->ShuffleTask();
  return Status::OK();
}

void MindRecordSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: MindRecordSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status MindRecordSamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sampler_name"] = "MindRecordSampler";
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
