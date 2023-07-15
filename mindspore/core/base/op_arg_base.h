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
#include <vector>
#include "base/base.h"
#include "ir/value.h"
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

  /// \brief Get the flatten shape vector, only supports simple data structure(Tensor, Scalar, Tuple/List (all elements
  /// must be Tensor and Scalar)).
  ///
  /// \return The flatten shape vector.
  /// For Tensor type, return its shape. For example, a Tensor with shape (8, 16), 'GetShape()' return
  /// std::vector<ShapeVector>{{8, 16}}.
  ///
  /// For Scalar type, return an std::vector<ShapeVector> containing an empty
  /// ShapeVector, i.e. std::vector<ShapeVector>{{}}.
  ///
  /// For Tuple/List (all elements must be Tensor and Scalar) type, the GetShape() return value
  /// consists of the shape of all elements in Typle/List. For example, if a Tuple of the structure ((8,16), (8,16))
  /// contains two Tensors of shape (8, 16), then the Tuple's GetShape() returns the value:
  /// std::vector<ShapeVector>{{8, 16}, {8, 16}}. A Tuple with a structure such as ((), ()) that contains two Scalar,
  /// the GetShape() of this Tuple returns the value std::vector<ShapeVector>{{}, {}}.
  virtual const std::vector<ShapeVector> &GetShape() = 0;

  /// \brief Get the object type of the OpArgBase.
  ///
  /// \return The object type of the OpArgBase.
  virtual TypePtr GetType() const = 0;

  /// \brief Get the value of the OpArgBase.
  ///
  /// \return The value of the OpArgBase if exists, else return kValueAny.
  virtual ValuePtr GetValue() const = 0;

  /// \brief Get whether the OpArgBase represents a dynamic length sequence.
  ///
  /// \return True if the OpArgBase represents a dynamic length sequence, otherwise False.
  virtual bool dynamic_len() const { return false; }
};

using OpArgBasePtr = std::shared_ptr<OpArgBase>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_BASE_OP_ARG_BASE_H_
