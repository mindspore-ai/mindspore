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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

#include <algorithm>
#include <string>

namespace mindspore {
namespace dataset {
Status RandomAccessOp::GetNumRowsInDataset(int64_t *num) const {
  // The sampler base class itself does not compute it's own num_rows_ value.
  // Instead, this value is computed by the derived leaf op during it's own initialization
  // after it has interacted with it's storage layers.
  // Here, it is just a getter method to return the value.  However, it is invalid if there is
  // not a value set for this count, so generate a failure if that is the case.
  if (num == nullptr || num_rows_ == -1) {
    RETURN_STATUS_UNEXPECTED("Get num rows in Dataset failed, num_rows has not been set yet.");
  }
  (*num) = num_rows_;
  return Status::OK();
}

SamplerRT::SamplerRT(int64_t num_samples, int64_t samples_per_tensor)
    : num_rows_(0),
      num_samples_(num_samples),
      samples_per_tensor_(samples_per_tensor),
      col_desc_(nullptr),
      is_initialized(false) {}

Status SamplerRT::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  std::shared_ptr<SamplerRT> child_sampler;
  if (HasChildSampler()) {
    child_sampler = std::dynamic_pointer_cast<SamplerRT>(child_[0]);
    if (!child_sampler) {
      std::string err_msg("[Internal ERROR] Cannot handshake, child is not a sampler object.");
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // Handshake and init child first.
    RETURN_IF_NOT_OK(child_sampler->HandshakeRandomAccessOp(op));
  }

  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "RandomAccessOp init failed, as it is nullptr.");

  // If there's a child sampler, set the row count to be it's sample count
  if (HasChildSampler()) {
    num_rows_ = child_sampler->num_samples_;
  } else {
    RETURN_IF_NOT_OK(op->GetNumRowsInDataset(&num_rows_));
  }

  // It's up to the derived class to check the validity of the two args
  // Because some sampler only needs one of the arg (weighted_random_sampler)
  RETURN_IF_NOT_OK(InitSampler());  // init sampler after callback

  return Status::OK();
}

Status SamplerRT::CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements) {
  if (col_desc_ == nullptr) {
    // a ColDescriptor for Tensor that holds SampleIds
    col_desc_ = std::make_unique<ColDescriptor>("sampleIds", DataType(DataType::DE_INT64), TensorImpl::kFlexible, 1);
  }
  TensorShape shape(std::vector<dsize_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, col_desc_->Type(), sample_ids));
  return Status::OK();
}

void SamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  // Sampler printing is usually only called in the show_all mode.
  // Derived classes will display the name, then call back to this base
  // for common info.
  // No-op in the summary mode.
  if (show_all) {
    out << "\nnum_rows_: " << num_rows_ << "\nnum_samples_: " << num_samples_;
  }
}

#ifdef ENABLE_PYTHON
Status SamplerRT::GetAllIdsThenReset(py::array *data) {
  std::shared_ptr<Tensor> sample_ids;
  TensorRow sample_row;

  // Get the only tensor inside the row that contains the actual SampleIds for the entire epoch
  RETURN_IF_NOT_OK(GetNextSample(&sample_row));
  sample_ids = sample_row[0];

  // check this tensorRow is not a ctrl tensorRow
  CHECK_FAIL_RETURN_UNEXPECTED(sample_row.Flags() == TensorRow::kFlagNone, "[Internal ERROR] ctrl row received.");

  // perform error checking! Next TensorRow supposed to be EOE since last one already contains all ids for current epoch
  RETURN_IF_NOT_OK(GetNextSample(&sample_row));
  CHECK_FAIL_RETURN_UNEXPECTED(sample_row.eoe(), "[Internal ERROR] Non EOE received in the end of epoch.");
  // Reset Sampler since this is the end of the epoch
  RETURN_IF_NOT_OK(ResetSampler());

  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      RETURN_IF_NOT_OK(sample_ids->GetDataAsNumpy(data));
    } catch (const std::runtime_error &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    }
  }
  return Status::OK();
}
#endif

Status SamplerRT::SetNumSamples(int64_t num_samples) {
  CHECK_FAIL_RETURN_UNEXPECTED(num_samples >= 0, "Invalid parameter, num_samples must be greater than or equal to 0.");
  num_samples_ = num_samples;
  return Status::OK();
}

int64_t SamplerRT::GetNumSamples() const { return num_samples_; }

int64_t SamplerRT::CalculateNumSamples(int64_t num_rows) {
  int64_t child_num_rows = num_rows;
  if (!child_.empty()) {
    child_num_rows = child_[0]->CalculateNumSamples(num_rows);
    // return -1 if child_num_rows is undetermined
    if (child_num_rows == -1) return child_num_rows;
  }

  return (num_samples_ > 0) ? std::min(child_num_rows, num_samples_) : child_num_rows;
}

Status SamplerRT::SetNumRowsInDataset(int64_t num_rows) {
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows > 0,
    "Invalid data, data rows of input dataset must not be less than or equal to 0, please check the input dataset.");
  num_rows_ = num_rows;
  return Status::OK();
}

Status SamplerRT::AddChild(std::shared_ptr<SamplerRT> child) {
  if (child == nullptr) {
    return Status::OK();
  }

  // Only samplers can be added, not any other DatasetOp.
  std::shared_ptr<SamplerRT> sampler = std::dynamic_pointer_cast<SamplerRT>(child);
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

  return Status::OK();
}

bool SamplerRT::HasChildSampler() const { return !child_.empty(); }

Status SamplerRT::GetAssociatedChildId(int64_t *out_associated_id, int64_t id) {
  if (child_ids_.empty()) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Trying to get associated child id, but there are no child ids!");
  }

  std::shared_ptr<Tensor> sample_ids = child_ids_[0];
  RETURN_IF_NOT_OK(sample_ids->GetItemAt<int64_t>(out_associated_id, {id}));
  return Status::OK();
}
Status SamplerRT::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_samples"] = num_samples_;
  if (this->HasChildSampler()) {
    std::vector<nlohmann::json> children_args;
    for (const auto &child : child_) {
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
