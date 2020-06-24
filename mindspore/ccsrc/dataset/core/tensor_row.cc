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

#include <utility>

#include "dataset/core/tensor_row.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {

TensorRow::TensorRow() noexcept : id_(kDefaultRowId) {}

TensorRow::TensorRow(size_type n, TensorRow::value_type t) noexcept : id_(kDefaultRowId), row_(n, t) {}

TensorRow::TensorRow(const TensorRow::vector_type &v) : id_(kDefaultRowId), row_(v) {}

TensorRow::TensorRow(row_id_type id, const std::initializer_list<value_type> &lst) : id_(id), row_(lst) {}

TensorRow::TensorRow(const TensorRow &tr) : id_(tr.id_), row_(tr.row_) {}

TensorRow &TensorRow::operator=(const TensorRow &tr) {
  if (this == &tr) {
    return *this;
  }
  row_ = tr.row_;
  id_ = tr.id_;
  return *this;
}

TensorRow &TensorRow::operator=(const std::initializer_list<TensorRow::value_type> &lst) {
  row_ = lst;
  return *this;
}

TensorRow::TensorRow(TensorRow::vector_type &&v) noexcept : id_(kDefaultRowId), row_(std::move(v)) {}

TensorRow::TensorRow(row_id_type id, std::initializer_list<value_type> &&lst) noexcept
    : id_(id), row_(std::move(lst)) {}

TensorRow::TensorRow(TensorRow &&tr) noexcept {
  id_ = tr.id_;
  row_ = std::move(tr.row_);
}

TensorRow &TensorRow::operator=(TensorRow &&tr) noexcept {
  if (this == &tr) {
    return *this;
  }
  row_ = std::move(tr.row_);
  id_ = tr.id_;
  tr.id_ = kDefaultRowId;
  return *this;
}

TensorRow &TensorRow::operator=(std::initializer_list<TensorRow::value_type> &&lst) noexcept {
  row_ = std::move(lst);
  return *this;
}

}  // namespace dataset
}  // namespace mindspore
