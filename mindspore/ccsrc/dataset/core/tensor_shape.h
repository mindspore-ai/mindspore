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
#ifndef DATASET_CORE_TENSOR_SHAPE_H_
#define DATASET_CORE_TENSOR_SHAPE_H_

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pybind11/pybind11.h"

#include "dataset/core/constants.h"
#include "dataset/core/global_context.h"
#include "dataset/util/allocator.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
// Class that represents a shape of a Tensor. A shape can be:
// -# Known shape (mKnown = true)
//        -# Scalar --> empty vector        --> <>
//        -# n-Dim  --> not empty vector    --> <d1, d2, d2, d3, ...> where di is >= 0\n
//           Example: <1,2>,  <1>,   <1,13,10,11,1>
// -# Unknown shape (mKnown = false)
//        -# Rank is unknown            --> empty vector     --> <>
//        -# one or more dim is unknown --> not empty vector --> <d1, d2, d2, d3, ...> where di is unknown\n
//           Example: <3,?> (the 1st dim is unknown)\n
//              <2,?,?,?> (all dims but the 0th dim are unknown)
//  TensorShape supports any dim > 0 and < 2^31-1
class TensorShape {
 public:
  static constexpr dsize_t kDimUnknown = -1;  // constant for an unknown dimension

  // Force the compiler to not create a no-arg constructor
  TensorShape() = delete;

  // Create a Shape from an initialization list (e.g., TensorShape s = {2,2}).
  // If one of the dims is set to DIM_UNKNOWN, the shape will flagged as unKnown
  // @param list
  explicit TensorShape(const std::initializer_list<dsize_t> &list);

  // Create a Shape from a vector (e.g., TensorShape s = std::vector<dsize_t>({2,2}) ).
  // If one of the dims is set to DIM_UNKNOWN, the shape will flagged as unKnown
  // @param list
  explicit TensorShape(const std::vector<dsize_t> &list);

  // Copy constructor
  // @param shape
  TensorShape(const TensorShape &shape);

  ~TensorShape() = default;

  // Create a scalar Shape (i.e., empty shape with mKnown = true)
  // @return TensorShape
  static TensorShape CreateScalar() { return TensorShape({}); }

  // Create a shape with an unknown rank.
  // @return TensorShape
  static TensorShape CreateUnknownRankShape();

  // Create a shape with a known rank .
  // @return TensorShape
  static TensorShape CreateUnknownShapeWithRank(dsize_t rank);

  // Insert a new dim into a copy of the current shape.
  // @param dim to be added
  // @param axis the index where dim should be added
  // @return New modified shape
  TensorShape InsertDim(dsize_t axis, dsize_t dim) const;

  // Insert new dim at index 0. For example,  <2,4> --> PrependDim(4) --> <4,2,4>
  // @param dim
  // @return
  TensorShape PrependDim(dsize_t dim) const;

  // Insert a new dim at the end of the shape. For example,  <2,4> --> PrependDim(4) --> <2,4,4>
  // @param dim
  // @return
  TensorShape AppendDim(dsize_t dim) const;

  // Create a shape based on OpenCV shape and type
  // @param cv_size
  // @param type int that represent the type in OpenCV, example CV_8U, CV_64S
  TensorShape(cv::MatSize cv_size, uint32_t type);

  dsize_t Size() const { return raw_shape_.size(); }

  dsize_t Rank() const { return raw_shape_.size(); }

  bool known() const { return known_; }

  bool empty() const { return raw_shape_.empty(); }

  dsize_t NumOfElements() const;

  bool operator==(const TensorShape &rhs) const { return known_ == rhs.known_ && raw_shape_ == rhs.raw_shape_; }

  bool operator!=(const TensorShape &rhs) const { return !(rhs == *this); }

  dsize_t operator[](const dsize_t index) const { return raw_shape_[index]; }

  // Return the Shape as a vector
  // @return
  std::vector<dsize_t> AsVector() const;

  // Returns the class info as a string
  // @return
  std::string ToString() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // Actual print function used by operator<<
  // @param out output string stream
  void Print(std::ostream &out) const;

  // << Stream output operator overload
  // @notes This allows you to print the info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param rO - reference to the TensorShape to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const TensorShape &so) {
    so.Print(out);
    return out;
  }

  explicit TensorShape(py::list l);

  py::list AsPyList();

  // Checks if the given index is a valid index for this tensor.
  // For example: Tensor<3,4> Index<1,1> is valid. But Index<4,1> or <1> are not.
  // @param index
  // @return bool
  bool IsValidIndex(const std::vector<dsize_t> &index) const;

  TensorShape Squeeze() const;

 private:
  // True if known and valid shape, false otherwise
  bool known_;
  // Vector to keep the dims of the shape.
  std::vector<dsize_t, IntAlloc> raw_shape_;

  // Internal utility function to iterate over a list, check if the dim is valid and then insert it into the shape.
  // @tparam T list
  // @param list Iterable list
  // @return true if the shape is valid and no overflow would be generated when counting the number of elements.
  //         False otherwise.
  template <typename T>
  void AddListToShape(const T &list);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_CORE_TENSOR_SHAPE_H_
