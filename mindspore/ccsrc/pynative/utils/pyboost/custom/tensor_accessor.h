/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_TENSOR_ACCESSOR_H_
#define MINDSPORE_CCSRC_EXTENSION_TENSOR_ACCESSOR_H_
#include <memory>
#include <vector>
#include "mindapi/base/shape_vector.h"
#include "utils/log_adapter.h"

namespace ms {
// Implementation of TensorAccessor template class
template <typename T, size_t N>
class TensorAccessor {
  static_assert(N > 1, "TensorAccessor is only valid for tensor with dimensions greater than or equal to 1.");

 public:
  // Constructor: Initializes the accessor with data pointer, shape, and shared_ptr to stride vector
  TensorAccessor(T *data, const int64_t *shape, const std::shared_ptr<ShapeVector> &stride_vec)
      : data_(data),
        shape_(shape),
        stride_vec_(stride_vec),
        stride_(stride_vec_->data() + (stride_vec_->size() - N)) {  // Calculate the current stride start
    MS_EXCEPTION_IF_NULL(shape);
    MS_EXCEPTION_IF_NULL(stride_vec_);
  }

  // Access tensor elements using multi-dimensional indices
  template <typename... Indices>
  T &operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == N, "Incorrect number of indices for TensorAccessor");
    return data_[ComputeOffset({static_cast<size_t>(indices)...})];
  }

  // Access a sub-tensor (reduce dimension recursively)
  TensorAccessor<T, N - 1> operator[](size_t index) const {
    MS_EXCEPTION_IF_CHECK_FAIL(index < shape_[0], "Index " + std::to_string(index) +
                                                    " is out of bounds for dimension with size " +
                                                    std::to_string(shape_[0]) + ".");
    return TensorAccessor<T, N - 1>(data_ + index * static_cast<size_t>(stride_[0]), shape_ + 1, stride_vec_);
  }

  T *data_ptr() const { return data_; }

  bool is_contiguous() const {
    for (size_t i = 0; i < N - 1; ++i) {
      if (stride_[i] != shape_[i + 1] * stride_[i + 1]) {
        return false;
      }
    }
    return true;
  }

 private:
  // Compute the linear offset for a multi-dimensional index
  size_t ComputeOffset(const std::vector<size_t> &indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < N; i++) {
      offset += indices[i] * static_cast<size_t>(stride_[i]);
    }
    return offset;
  }

  T *data_;                                  // Pointer to the tensor data
  const int64_t *shape_;                     // Pointer to the shape array
  std::shared_ptr<ShapeVector> stride_vec_;  // Shared pointer to the stride vector
  const int64_t *stride_;                    // Pointer to the current stride start
};

// Specialization of TensorAccessor for 1-dimensional tensors
template <typename T>
class TensorAccessor<T, 1> {
 public:
  // Constructor: Initializes the accessor with data pointer, shape, and shared_ptr to stride vector
  TensorAccessor(T *data, const int64_t *shape, const std::shared_ptr<ShapeVector> &stride_vec)
      : data_(data), shape_(shape), stride_vec_(stride_vec) {
    MS_EXCEPTION_IF_NULL(shape);
    MS_EXCEPTION_IF_NULL(stride_vec_);
  }

  // Access elements in the tensor
  T &operator[](size_t index) const {
    MS_EXCEPTION_IF_CHECK_FAIL(
      index < shape_[0],
      "Index " + std::to_string(index) + " is out of bounds for a tensor with size " + std::to_string(shape_[0]) + ".");
    return data_[index * static_cast<size_t>(stride_vec_->back())];
  }

  T *data_ptr() const { return data_; }
  bool is_contiguous() const { return true; }

 private:
  T *data_;                                  // Pointer to the tensor data
  const int64_t *shape_;                     // Pointer to the shape array
  std::shared_ptr<ShapeVector> stride_vec_;  // Shared pointer to the stride vector
};
}  // namespace ms
#endif  // MINDSPORE_CCSRC_EXTENSION_TENSOR_ACCESSOR_H_
