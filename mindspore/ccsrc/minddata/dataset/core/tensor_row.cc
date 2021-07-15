/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/core/tensor_row.h"

#include <utility>

namespace mindspore {
namespace dataset {

TensorRow::TensorRow() noexcept : id_(kDefaultRowId), path_({}), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(size_type n, const TensorRow::value_type &t) noexcept
    : id_(kDefaultRowId), path_({}), row_(n, t), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(const TensorRow::vector_type &v)
    : id_(kDefaultRowId), path_({}), row_(v), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(row_id_type id, const std::initializer_list<value_type> &lst)
    : id_(id), path_({}), row_(lst), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(const TensorRow &tr)
    : id_(tr.id_), path_(tr.path_), row_(tr.row_), tensor_row_flag_(tr.tensor_row_flag_) {}

TensorRow::TensorRow(TensorRow::TensorRowFlags flag) : id_(kDefaultRowId), path_({}), tensor_row_flag_(flag) {}

TensorRow &TensorRow::operator=(const TensorRow &tr) {
  if (this == &tr) {
    return *this;
  }
  row_ = tr.row_;
  id_ = tr.id_;
  path_ = tr.path_;
  tensor_row_flag_ = tr.tensor_row_flag_;
  return *this;
}

TensorRow &TensorRow::operator=(const std::initializer_list<TensorRow::value_type> &lst) {
  row_ = lst;
  tensor_row_flag_ = kFlagNone;
  return *this;
}

TensorRow::TensorRow(TensorRow::vector_type &&v) noexcept
    : id_(kDefaultRowId), path_({}), row_(std::move(v)), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(row_id_type id, std::initializer_list<value_type> &&lst) noexcept
    : id_(id), path_({}), row_(std::move(lst)), tensor_row_flag_(kFlagNone) {}

TensorRow::TensorRow(TensorRow &&tr) noexcept {
  id_ = tr.id_;
  path_ = std::move(tr.path_);
  row_ = std::move(tr.row_);
  tensor_row_flag_ = tr.tensor_row_flag_;
}

TensorRow &TensorRow::operator=(TensorRow &&tr) noexcept {
  if (this == &tr) {
    return *this;
  }
  row_ = std::move(tr.row_);
  id_ = tr.id_;
  tr.id_ = kDefaultRowId;
  path_ = std::move(tr.path_);
  tensor_row_flag_ = tr.tensor_row_flag_;
  return *this;
}

TensorRow &TensorRow::operator=(std::initializer_list<TensorRow::value_type> &&lst) noexcept {
  row_ = std::move(lst);
  tensor_row_flag_ = kFlagNone;
  return *this;
}

Status TensorRow::ValidateTensorRow(const TensorRow &input, const DataType &data_type) {
  if (data_type == DataType::DE_UNKNOWN) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type was not recognized.");
  }
  if (data_type == DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type string is not supported.");
  }
  if (input.size() != 1) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input TensorRow must have exactly one tensor.");
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
