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
#include "dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
Sampler::Sampler(int64_t samples_per_buffer)
    : DatasetOp(0), num_rows_(0), num_samples_(0), samples_per_buffer_(samples_per_buffer), col_desc_(nullptr) {}

Status Sampler::Init(const RandomAccessOp *op) {
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr && samples_per_buffer_ > 0, "Fail to init Sampler()\n");
  RETURN_IF_NOT_OK(op->GetNumSamples(&num_samples_));
  RETURN_IF_NOT_OK(op->GetNumRowsInDataset(&num_rows_));
  // It's up to the derived class to check the validity of the two args
  // Because some sampler only needs one of the arg (weighted_random_sampler)
  return Status::OK();
}

Status Sampler::CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements) {
  if (num_elements == 0) {
    RETURN_STATUS_UNEXPECTED("num of Elements is 0");
  }
  if (col_desc_ == nullptr) {
    // a ColDescriptor for Tensor that holds SampleIds
    col_desc_ = make_unique<ColDescriptor>("sampleIds", DataType(DataType::DE_INT64), TensorImpl::kFlexible, 1);
  }
  TensorShape shape(std::vector<dsize_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateTensor(sample_ids, col_desc_->tensorImpl(), shape, col_desc_->type()));
  (void)(*sample_ids)->StartAddr();  // allocate memory in case user forgets!
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
