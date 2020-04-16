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

Status Sampler::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "RandomAccessOp is nullptr\n");
  RETURN_IF_NOT_OK(op->GetNumSamples(&num_samples_));
  RETURN_IF_NOT_OK(op->GetNumRowsInDataset(&num_rows_));
  // It's up to the derived class to check the validity of the two args
  // Because some sampler only needs one of the arg (weighted_random_sampler)
  RETURN_IF_NOT_OK(InitSampler());  // init sampler after callback
  return Status::OK();
}

Status Sampler::CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements) {
  if (num_elements == 0) {
    RETURN_STATUS_UNEXPECTED("num of Elements is 0");
  }
  if (col_desc_ == nullptr) {
    // a ColDescriptor for Tensor that holds SampleIds
    col_desc_ = std::make_unique<ColDescriptor>("sampleIds", DataType(DataType::DE_INT64), TensorImpl::kFlexible, 1);
  }
  TensorShape shape(std::vector<dsize_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateTensor(sample_ids, col_desc_->tensorImpl(), shape, col_desc_->type()));
  (void)(*sample_ids)->StartAddr();  // allocate memory in case user forgets!
  return Status::OK();
}

Status Sampler::GetAllIdsThenReset(py::array *data) {
  std::unique_ptr<DataBuffer> db;
  std::shared_ptr<Tensor> sample_ids;

  // check samples_per_buffer is properly set and doesn't overflow
  CHECK_FAIL_RETURN_UNEXPECTED(samples_per_buffer_ + 1 > 1, "samples_per_buffer invalid");

  // A call to derived class to get sample ids wrapped inside a buffer
  RETURN_IF_NOT_OK(GetNextBuffer(&db));
  // Get the only tensor inside the buffer that contains the actual SampleIds for the entire epoch
  RETURN_IF_NOT_OK(db->GetTensor(&sample_ids, 0, 0));
  // check this buffer is not a ctrl buffer
  CHECK_FAIL_RETURN_UNEXPECTED(db->buffer_flags() == DataBuffer::kDeBFlagNone, "ERROR ctrl buffer received");
  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      RETURN_IF_NOT_OK(sample_ids->GetDataAsNumpy(data));
    } catch (const std::runtime_error &e) {
      return Status(StatusCode::kPyFuncException, e.what());
    }
  }
  // perform error checking! Next buffer supposed to be EOE since last one already contains all ids for current epoch
  RETURN_IF_NOT_OK(GetNextBuffer(&db));
  CHECK_FAIL_RETURN_UNEXPECTED(db->eoe(), "ERROR Non EOE received");
  // Reset Sampler since this is the end of the epoch
  RETURN_IF_NOT_OK(Reset());
  return Status::OK();
}

Status Sampler::SetNumSamples(int64_t num_samples) {
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples > 0, "num_samples is negative or 0");
  num_samples_ = num_samples;
  return Status::OK();
}

Status Sampler::SetNumRowsInDataset(int64_t num_rows) {
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows > 0, "num_rows is negative or 0");
  num_rows_ = num_rows;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
