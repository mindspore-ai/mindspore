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

#include <string>

namespace mindspore {
namespace dataset {
Sampler::Sampler(int64_t samples_per_buffer)
    : DatasetOp(0), num_rows_(0), num_samples_(0), samples_per_buffer_(samples_per_buffer), col_desc_(nullptr) {}

Status Sampler::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  std::shared_ptr<Sampler> child_sampler;
  if (HasChildSampler()) {
    child_sampler = std::dynamic_pointer_cast<Sampler>(child_[0]);
    if (!child_sampler) {
      std::string err_msg("Cannot handshake, child is not a sampler object.");
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // Handshake and init child first.
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_sampler->HandshakeRandomAccessOp(op));
    }
  }

  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "RandomAccessOp is nullptr\n");
  RETURN_IF_NOT_OK(op->GetNumSamples(&num_samples_));
  if (HasChildSampler()) {
    int64_t child_num_samples = child_sampler->num_samples();
    num_rows_ = child_num_samples;
  } else {
    RETURN_IF_NOT_OK(op->GetNumRowsInDataset(&num_rows_));
  }

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
  RETURN_IF_NOT_OK(
    (*sample_ids)->AllocateBuffer((*sample_ids)->SizeInBytes()));  // allocate memory in case user forgets!
  return Status::OK();
}

void Sampler::Print(std::ostream &out, bool show_all) const {
  out << "(sampler): base\n";

  if (show_all) {
    out << "num_rows_: " << num_rows_ << '\n';
    out << "num_samples_: " << num_samples_ << '\n';
  }
}

Status Sampler::GetAllIdsThenReset(py::array *data) {
  std::unique_ptr<DataBuffer> db;
  std::shared_ptr<Tensor> sample_ids;

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

Status Sampler::AddChild(std::shared_ptr<DatasetOp> child) {
  if (child == nullptr) {
    return Status::OK();
  }

  // Only samplers can be added, not any other DatasetOp.
  std::shared_ptr<Sampler> sampler = std::dynamic_pointer_cast<Sampler>(child);
  if (!sampler) {
    std::string err_msg("Cannot add child, child is not a sampler object.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Samplers can have at most 1 child.
  if (!child_.empty()) {
    std::string err_msg("Cannot add child sampler, this sampler already has a child.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  child_.push_back(child);

  // doesn't work, protected?
  // child->AddParent(this);
  return Status::OK();
}

bool Sampler::HasChildSampler() { return !child_.empty(); }

Status Sampler::GetAssociatedChildId(int64_t *out_associated_id, int64_t id) {
  if (child_ids_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Trying to get associated child id, but there are no child ids!");
  }

  TensorRow sample_row;
  RETURN_IF_NOT_OK(child_ids_->GetRow(0, &sample_row));
  std::shared_ptr<Tensor> sample_ids = sample_row[0];
  RETURN_IF_NOT_OK(sample_ids->GetItemAt<int64_t>(out_associated_id, {id}));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
