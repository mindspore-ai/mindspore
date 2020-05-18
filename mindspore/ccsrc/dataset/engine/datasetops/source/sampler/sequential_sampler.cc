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

#include <memory>

namespace mindspore {
namespace dataset {
SequentialSampler::SequentialSampler(int64_t samples_per_buffer) : Sampler(samples_per_buffer), next_id_(0) {}

Status SequentialSampler::GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) {
  if (next_id_ > num_samples_) {
    RETURN_STATUS_UNEXPECTED("Sequential Sampler Internal Error");
  } else if (next_id_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    (*out_buffer) = std::make_unique<DataBuffer>(next_id_, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> sampleIds;
    int64_t lastId = (samples_per_buffer_ + next_id_ > num_samples_) ? num_samples_ : samples_per_buffer_ + next_id_;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, lastId - next_id_));
    int64_t *idPtr = reinterpret_cast<int64_t *>(sampleIds->GetMutableBuffer());
    while (next_id_ < lastId) {
      *(idPtr++) = next_id_++;
    }
    TensorRow row(1, sampleIds);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
  }
  return Status::OK();
}

Status SequentialSampler::InitSampler() {
  num_samples_ = (num_samples_ <= 0) ? num_rows_ : num_samples_;  // if num_samples < 0, try if num_rows is set
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0 && samples_per_buffer_ > 0, "Fail to init Sequential Sampler");
  samples_per_buffer_ = samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;
  return Status::OK();
}

Status SequentialSampler::Reset() {
  CHECK_FAIL_RETURN_UNEXPECTED(next_id_ == num_samples_, "ERROR Reset() called early/late");
  next_id_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
