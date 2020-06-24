/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "dataset/engine/datasetops/bucket_batch_by_length_op.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "dataset/core/pybind_support.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/engine/opt/pass.h"
#include "dataset/util/status.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
BucketBatchByLengthOp::Builder::Builder(std::vector<std::string> length_dependent_columns,
                                        std::vector<int32_t> bucket_boundaries, std::vector<int32_t> bucket_batch_sizes)
    : builder_length_dependent_columns_(length_dependent_columns),
      builder_bucket_boundaries_(bucket_boundaries),
      builder_bucket_batch_sizes_(bucket_batch_sizes),
      builder_pad_info_({}),
      builder_pad_to_bucket_boundary_(false),
      builder_drop_remainder_(false) {
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  builder_op_connector_size_ = config_manager->op_connector_size();
}

Status BucketBatchByLengthOp::Builder::SanityCheck() {
  std::string error_message;

  if (builder_length_dependent_columns_.empty()) {
    error_message += "At least 1 column must be specified for element length calculation.\n";
  }

  if (builder_bucket_boundaries_.empty()) {
    error_message += "At least 1 bucket boundary must be specified.\n";
  }

  if (builder_bucket_batch_sizes_.size() != builder_bucket_boundaries_.size() + 1) {
    error_message += "There must be exactly one bucket batch size specified for each bucket boundary.\n";
  }

  CHECK_FAIL_RETURN_UNEXPECTED(error_message.empty(), error_message);

  return Status::OK();
}

Status BucketBatchByLengthOp::Builder::Build(std::shared_ptr<BucketBatchByLengthOp> *new_bucket_batch_by_length_op) {
  RETURN_IF_NOT_OK(SanityCheck());

  // insert 0 for the first bucket
  builder_bucket_boundaries_.insert(builder_bucket_boundaries_.begin(), 0);

  *new_bucket_batch_by_length_op = std::make_shared<BucketBatchByLengthOp>(
    builder_length_dependent_columns_, builder_bucket_boundaries_, builder_bucket_batch_sizes_,
    builder_element_length_function_, builder_pad_info_, builder_pad_to_bucket_boundary_, builder_drop_remainder_,
    builder_op_connector_size_);

  return Status::OK();
}

BucketBatchByLengthOp::BucketBatchByLengthOp(std::vector<std::string> length_dependent_columns,
                                             std::vector<int32_t> bucket_boundaries,
                                             std::vector<int32_t> bucket_batch_sizes,
                                             py::function element_length_function, PadInfo pad_info,
                                             bool pad_to_bucket_boundary, bool drop_remainder,
                                             int32_t op_connector_size)
    : PipelineOp(op_connector_size),
      length_dependent_columns_(length_dependent_columns),
      bucket_boundaries_(bucket_boundaries),
      bucket_batch_sizes_(bucket_batch_sizes),
      element_length_function_(element_length_function),
      pad_info_(pad_info),
      pad_to_bucket_boundary_(pad_to_bucket_boundary),
      drop_remainder_(drop_remainder),
      batch_count_(0) {
  for (int i = 0; i < bucket_batch_sizes_.size(); i++) {
    buckets_.push_back(std::make_unique<TensorQTable>());
  }
}

Status BucketBatchByLengthOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

void BucketBatchByLengthOp::Print(std::ostream &out, bool show_all) const { out << "BucketBatchByLengthOp\n"; }

Status BucketBatchByLengthOp::operator()() {
  TaskManager::FindMe()->Post();

  TensorRow current_row;
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
  RETURN_IF_NOT_OK(AssignColMapFromChild());
  while (!child_iterator_->eof_handled()) {
    while (!current_row.empty()) {
      int32_t element_length;
      RETURN_IF_NOT_OK(ObtainElementLength(&element_length, current_row));

      int bucket_index = bucket_boundaries_.size() - 1;
      while (element_length < bucket_boundaries_[bucket_index]) {
        bucket_index--;
      }

      buckets_[bucket_index]->push_back(current_row);

      if (buckets_[bucket_index]->size() == bucket_batch_sizes_[bucket_index]) {
        RETURN_IF_NOT_OK(PadAndBatchBucket(bucket_index, bucket_batch_sizes_[bucket_index]));
      }

      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
    }

    // got EOE, do what we need to do with remainders in each bucket
    if (!drop_remainder_) {
      for (int i = 0; i < bucket_boundaries_.size(); i++) {
        if (!buckets_[i]->empty()) {
          RETURN_IF_NOT_OK(PadAndBatchBucket(i, buckets_[i]->size()));
        }
      }
    }

    // need to send EOE manually since we set state to idle in EoeRecieved()
    std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));

    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
  }

  return Status::OK();
}

Status BucketBatchByLengthOp::ObtainElementLength(int32_t *out_element_length, TensorRow element) {
  // call pyfunc here if given pyfunc, otherwise return 0th dimension of shape of
  // the single column specified in length_dependent_columns_
  if (element_length_function_) {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      size_t number_of_arguments = length_dependent_columns_.size();
      py::tuple input_arguments(number_of_arguments);
      for (size_t i = 0; i < number_of_arguments; i++) {
        py::array argument_value;
        int32_t column_index = column_name_id_map_[length_dependent_columns_[i]];
        RETURN_IF_NOT_OK(element[column_index]->GetDataAsNumpy(&argument_value));
        input_arguments[i] = argument_value;
      }

      py::object length = element_length_function_(*input_arguments);
      *out_element_length = length.cast<int32_t>();
      if (*out_element_length < 0) {
        return Status(StatusCode::kPyFuncException, "Element length function should return a non negative integer.");
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kPyFuncException, e.what());
    } catch (const py::cast_error &e) {
      return Status(StatusCode::kPyFuncException, "Count not cast output of element length function to int32_t.");
    }
  } else {
    *out_element_length = element[0]->shape()[0];
  }

  return Status::OK();
}

Status BucketBatchByLengthOp::PadAndBatchBucket(int32_t bucket_index, int32_t batch_size) {
  std::unique_ptr<TensorQTable> *bucket = &buckets_[bucket_index];

  PadInfo pad_info_copy = pad_info_;
  if (pad_to_bucket_boundary_) {
    for (auto &pair : pad_info_copy) {
      std::vector<dsize_t> pad_shape = pair.second.first.AsVector();

      for (size_t i = 0; i < pad_shape.size(); i++) {
        if (pad_shape[i] == TensorShape::kDimUnknown) {
          if (bucket_index + 1 >= bucket_boundaries_.size()) {
            std::string error_message = "Requested to pad to bucket boundary, element falls in last bucket";
            return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, error_message);
          }

          pad_shape[i] = bucket_boundaries_[bucket_index + 1] - 1;
        }
      }

      pair.second.first = TensorShape(pad_shape);
    }
  }

  // PadColumns will change the data in bucket
  RETURN_IF_NOT_OK(BatchOp::PadColumns(bucket, pad_info_copy, column_name_id_map_));

  std::unique_ptr<TensorQTable> batched_bucket = std::make_unique<TensorQTable>();
  RETURN_IF_NOT_OK(BatchOp::BatchRows(bucket, &batched_bucket, batch_size));
  (*bucket)->clear();

  std::unique_ptr<DataBuffer> batched_buffer = std::make_unique<DataBuffer>(batch_count_, DataBuffer::kDeBFlagNone);
  batched_buffer->set_tensor_table(std::move(batched_bucket));
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(batched_buffer)));

  batch_count_++;

  return Status::OK();
}

Status BucketBatchByLengthOp::Reset() {
  batch_count_ = 0;

  for (int i = 0; i < buckets_.size(); i++) {
    buckets_[i] = std::make_unique<TensorQTable>();
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
