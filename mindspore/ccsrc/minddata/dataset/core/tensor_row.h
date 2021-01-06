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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_ROW_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_ROW_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {

class TensorRow;                             // A set of Tensor pointers with an id
using TensorTable = std::vector<TensorRow>;  // The table of tensors is a vector of rows
using TensorQTable = std::deque<TensorRow>;  // A different flavour of tensor table, this one has queue functionality

class TensorRow {
 public:
  static constexpr row_id_type kDefaultRowId = -1;  // Default row id

  // Type definitions
  using size_type = dsize_t;
  using value_type = std::shared_ptr<Tensor>;
  using reference = std::shared_ptr<Tensor> &;
  using const_reference = const std::shared_ptr<Tensor> &;
  using vector_type = std::vector<std::shared_ptr<Tensor>>;
  using iterator = std::vector<std::shared_ptr<Tensor>>::iterator;
  using const_iterator = std::vector<std::shared_ptr<Tensor>>::const_iterator;

  TensorRow() noexcept;

  TensorRow(size_type n, value_type t) noexcept;

  // Copy Constructors
  explicit TensorRow(const vector_type &v);

  TensorRow(row_id_type id, const std::initializer_list<value_type> &lst);

  TensorRow(const TensorRow &tr);

  TensorRow &operator=(const TensorRow &tr);

  TensorRow &operator=(const std::initializer_list<value_type> &lst);

  // Move Constructors
  explicit TensorRow(vector_type &&v) noexcept;

  TensorRow(row_id_type id, std::initializer_list<value_type> &&lst) noexcept;

  TensorRow(TensorRow &&tr) noexcept;

  TensorRow &operator=(TensorRow &&tr) noexcept;

  TensorRow &operator=(std::initializer_list<value_type> &&lst) noexcept;

  // Destructor
  ~TensorRow() = default;

  /// Convert a vector of primitive types to a TensorRow consisting of one 1-D Tensor with the shape n.
  /// \tparam `T`
  /// \param[in] o input vector
  /// \param[out] output TensorRow
  template <typename T>
  static Status ConvertToTensorRow(const std::vector<T> &o, TensorRow *output) {
    DataType data_type = DataType::FromCType<T>();
    if (data_type == DataType::DE_UNKNOWN) {
      RETURN_STATUS_UNEXPECTED("ConvertToTensorRow: Data type was not recognized.");
    }
    if (data_type == DataType::DE_STRING) {
      RETURN_STATUS_UNEXPECTED("ConvertToTensorRow: Data type string is not supported.");
    }
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(o, &tensor));
    output->push_back(tensor);
    return Status::OK();
  }

  /// Convert a single primitive type to a TensorRow consisting of one single data Tensor.
  /// \tparam `T`
  /// \param[in] o input
  /// \param[out] output TensorRow
  template <typename T>
  static Status ConvertToTensorRow(const T &o, TensorRow *output) {
    DataType data_type = DataType::FromCType<T>();
    if (data_type == DataType::DE_UNKNOWN) {
      RETURN_STATUS_UNEXPECTED("ConvertToTensorRow: Data type was not recognized.");
    }
    if (data_type == DataType::DE_STRING) {
      RETURN_STATUS_UNEXPECTED("ConvertToTensorRow: Data type string is not supported.");
    }
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(o, &tensor));
    output->push_back(tensor);
    return Status::OK();
  }

  /// Return the value in a TensorRow consisting of 1 single data Tensor.
  /// \tparam `T`
  /// \param[in] input TensorRow
  /// \param[out] o the primitive variable
  template <typename T>
  static Status ConvertFromTensorRow(const TensorRow &input, T *o) {
    DataType data_type = DataType::FromCType<T>();
    if (data_type == DataType::DE_UNKNOWN) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type was not recognized.");
    }
    if (data_type == DataType::DE_STRING) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type string is not supported.");
    }
    if (input.size() != 1) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input TensorRow must have exactly one tensor.");
    }
    if (input.at(0)->type() != data_type) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The output type doesn't match the input tensor type.");
    }
    if (input.at(0)->shape() != TensorShape({})) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input tensors must be a scalar tensor.");
    }
    return input.at(0)->GetItemAt(o, {0});
  }

  /// Convert a TensorRow consisting of one 1-D tensor to a vector of size n.
  /// \tparam `T`
  /// \param[in] o TensorRow consisting of one 1-D tensor
  /// \param[out] o vector of primitive variable
  template <typename T>
  static Status ConvertFromTensorRow(const TensorRow &input, std::vector<T> *o) {
    DataType data_type = DataType::FromCType<T>();
    if (data_type == DataType::DE_UNKNOWN) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type was not recognized.");
    }
    if (data_type == DataType::DE_STRING) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type string is not supported.");
    }
    if (input.size() != 1) {
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input TensorRow must have exactly one tensor.");
    }
    if (input.at(0)->Rank() != 1)
      RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input tensor must have a rank of 1.");
    for (auto it = input.at(0)->begin<T>(); it != input.at(0)->end<T>(); it++) {
      o->push_back(*it);
    }
    return Status::OK();
  }

  // Functions to fetch/set id/vector
  row_id_type getId() const { return id_; }

  void setId(row_id_type id) { id_ = id; }

  std::vector<std::string> getPath() const { return path_; }

  void setPath(std::vector<std::string> path) { path_ = path; }

  const vector_type &getRow() const { return row_; }

  int64_t SizeInBytes() const {
    size_t sz = 0;
    for (auto &it : row_) {
      sz += it->SizeInBytes();
    }
    return sz;
  }

  // Wrapper functions to support vector operations
  void emplace_back(value_type t) { row_.emplace_back(t); }

  void push_back(value_type t) { row_.push_back(t); }

  void clear() noexcept { row_.clear(); }

  size_type size() const noexcept { return row_.size(); }

  void reserve(size_type size) { row_.reserve(size); }

  void resize(size_type size) { row_.resize(size); }

  bool empty() { return row_.empty(); }

  void insert(iterator position, iterator first, iterator last) { row_.insert(position, first, last); }

  // Wrapper functions to support vector element access
  reference at(size_type index) { return row_.at(index); }

  const_reference at(size_type index) const { return row_.at(index); }

  reference front() { return row_.front(); }

  const_reference front() const { return row_.front(); }

  reference back() { return row_.back(); }

  const_reference back() const { return row_.back(); }

  reference operator[](size_type index) { return row_[index]; }

  const_reference operator[](size_type index) const { return row_[index]; }

  // Wrapper functions to support vector iteration
  iterator begin() { return row_.begin(); }

  const_iterator begin() const { return row_.begin(); }

  iterator end() { return row_.end(); }

  const_iterator end() const { return row_.end(); }

 protected:
  row_id_type id_;
  std::vector<std::string> path_;
  std::vector<std::shared_ptr<Tensor>> row_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_ROW_H_
