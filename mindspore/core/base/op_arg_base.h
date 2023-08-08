/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_OP_ARG_BASE_H_
#define MINDSPORE_CORE_BASE_OP_ARG_BASE_H_

#include <memory>
#include "base/base.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
/// \brief OpArgBase provides an abstract interface related to operator's arguments, which can obtain and save the shape
/// and type information of the input and output of operators.
class MS_CORE_API OpArgBase {
 public:
  /// \brief Default constructor of OpArgBase.
  OpArgBase() = default;

  /// \brief Default destructor of OpArgBase.
  ~OpArgBase() = default;

  /// \brief Get the flatten shape vector.
  ///
  /// \return The flatten shape vector.
  /// For Tensor type, return its shape.
  /// For Scalar type, return an empty ShapeVector.
  /// For simple Tuple/List type (where type and shape are exactly the same), shape vector is returned as the number of
  /// elements + the shape of the element. For example, a tuple like ((3,4),(3,4)) containing two Tensor, the shape of
  /// each Tensor is all same and is (3,4), the shape vector for the Tuple is (2,3,4) where 2 means the number of
  /// elements in the Tuple.
  /// For other types, the function should throw an exception that does not support obtaining a shape vector.
  virtual const ShapeVector &shape_vector() const = 0;

  /// \brief Set the flatten shape vector.
  ///
  /// \param[in] shape The flatten shape vector to be set.
  virtual void set_shape_vector(const ShapeVector &shape_vector) = 0;

  /// \brief Get the object type of the OpArgBase.
  ///
  /// \return The object type of the OpArgBase.
  virtual TypePtr type() const = 0;

  /// \brief Set the type for the OpArgBase.
  ///
  /// \param[in] type The type of OpArgBase to be set.
  virtual void set_type(const TypePtr &type) = 0;

  /// \brief Get whether the OpArgBase represents a dynamic length sequence.
  ///
  /// \return True if the OpArgBase represents a dynamic length sequence, otherwise False.
  virtual bool dynamic_len() const { return false; }
};

using OpArgBasePtr = std::shared_ptr<OpArgBase>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_BASE_OP_ARG_BASE_H_
