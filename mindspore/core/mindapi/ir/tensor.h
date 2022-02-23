/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_IR_TENSOR_H_
#define MINDSPORE_CORE_MINDAPI_IR_TENSOR_H_

#include <cstdint>
#include "mindapi/base/base.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"

namespace mindspore::api {
/// \brief Tensor represents a multi-dimensional array of elements.
class MIND_API Tensor : public Value {
 public:
  MIND_API_BASE_MEMBER(Tensor);

  /// \brief Create a lazy allocated tensor.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  Tensor(TypeId data_type, const ShapeVector &shape);

  /// \brief Create a tensor with input data buffer.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The input data to be copied into tensor.
  /// \param[in] data_len The length of data in bytes.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len);

  /// \brief Get the shape of the tensor.
  /// The shape of a tensor is stored in a vector<int64_t>. Each element of the
  /// vector represents the size of a dimension of the tensor. The order of each
  /// element in the vector is the same as the the dimension's order it represents.
  ///
  /// \return A vector<int64_t> which represents the shape of the tensor.
  const ShapeVector &shape() const;

  /// \brief Set the shape of tensor.
  ///
  /// \param[in] shape The shape to be set.
  void set_shape(const ShapeVector &shape);

  /// \brief Get the data type of the tensor.
  ///
  /// \return The data type of the tensor.
  TypeId data_type() const;

  /// \brief Set the data type of the tensor.
  ///
  /// \param[in] data_type The data type to be set.
  void set_data_type(const TypeId data_type);

  /// \brief Get The pointer to the underlying memory block for data storage.
  ///
  /// \return The pointer to the underlying data.
  const void *data() const;

  /// \brief Get The pointer to the underlying memory block for data storage.
  ///
  /// \return The pointer to the underlying data.
  void *data();

  /// \brief Get tensor data size.
  ///
  /// \return The total number of elements in the tensor.
  size_t DataSize() const;

  /// \brief Get tensor data size in bytes.
  ///
  /// \return The total number of bytes for the tensor data.
  std::size_t Size() const;
};

using TensorPtr = SharedPtr<Tensor>;
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_TENSOR_H_
