/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_IR_ABSTRACT_H_
#define MINDSPORE_CORE_MINDAPI_IR_ABSTRACT_H_

#include "mindapi/base/base.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/shape.h"
#include "mindapi/ir/type.h"
#include "mindapi/ir/value.h"

namespace mindspore::api {
/// \brief AbstractBase defines base interfaces for abstract of an anf node.
class MIND_API AbstractBase : public Base {
 public:
  MIND_API_BASE_MEMBER(AbstractBase);

  /// \brief Clone an abstract from this abstract.
  ///
  /// \return A pointer to the cloned abstract.
  AbstractBasePtr Clone() const;

  /// \brief Get the abstract type.
  ///
  /// \return A pointer to the Type.
  TypePtr type() const;

  /// \brief Get the abstract value.
  ///
  /// \return A pointer to the Value.
  ValuePtr value() const;

  /// \brief Set the type for this abstract.
  ///
  /// \param[in] type The type to be set.
  void set_type(const TypePtr &type);

  /// \brief Set the value for this abstract.
  ///
  /// \param[in] value The value to be set.
  void set_value(const ValuePtr &value);
};

/// \brief AbstractTensor describes a tensor's type, shape and value.
class MIND_API AbstractTensor : public AbstractBase {
 public:
  MIND_API_BASE_MEMBER(AbstractTensor);

  /// \brief Create AbstractTensor from the given type and shape.
  ///
  /// \param[in] type The data type id of the tensor.
  /// \param[in] shape The shape of the tensor.
  AbstractTensor(TypeId type, const ShapeVector &shape);

  /// \brief Get the element abstract.
  ///
  /// \return A pointer to the element abstract.
  AbstractBasePtr element() const;

  /// \brief Get the shape of the abstract.
  ///
  /// \return A pointer to the shape.
  ShapePtr shape() const;
};

using AbstractTensorPtr = SharedPtr<AbstractTensor>;

/// \brief AbstractSequence describes the abstract for a tuple or list.
class MIND_API AbstractSequence : public AbstractBase {
 public:
  MIND_API_BASE_MEMBER(AbstractSequence);

  /// \brief Get element abstracts.
  ///
  /// \return A vector of element abstracts.
  AbstractBasePtrList elements() const;
};

using AbstractSequencePtr = SharedPtr<AbstractSequence>;

/// \brief AbstractTuple describes the abstract for a tuple.
class MIND_API AbstractTuple : public AbstractSequence {
 public:
  MIND_API_BASE_MEMBER(AbstractTuple);
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_ABSTRACT_H_
