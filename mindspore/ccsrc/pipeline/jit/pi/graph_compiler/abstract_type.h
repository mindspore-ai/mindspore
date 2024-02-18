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
#ifndef MINDSPORE_PI_JIT_ABSTRACT_TYPE_H_
#define MINDSPORE_PI_JIT_ABSTRACT_TYPE_H_

#include <string>
#include <memory>
#include "mindspore/core/abstract/abstract_value.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/type.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;
class AbstractType : public ir::Type {
 public:
  /// \brief The constructor for AbstractType.
  explicit AbstractType(const abstract::AbstractBasePtr &abs, const py::object &obj = py::none())
      : ir::Type((abs == nullptr) ? 0 : abs->BuildType()->type_id(),
                 (abs == nullptr) ? "Unknown" : abs->BuildType()->ToString()),
        abs_(abs),
        obj_(obj) {}

  /// \brief The copy constructor of AbstractType.
  ///
  /// \param[in] other Define another instance of AbstractType.
  ///
  /// \return The instance of AbstractType.
  explicit AbstractType(const AbstractType &other) : Type(other), abs_(other.abs_) {}

  /// \brief Destructor.
  virtual ~AbstractType() = default;

  /// \brief The operator overloading for "==".
  ///
  /// \param[in] rhs Define the right operand of "==".
  ///
  /// \return The comparison result.
  virtual bool operator==(const AbstractType &rhs) { return this == &rhs || abs_ == rhs.abs_; }

  /// \brief Get the abstract member of the AbstractType.
  ///
  /// \return The abstract member of the object.
  const abstract::AbstractBasePtr &GetAbstract() const { return abs_; }

  /// \brief Get the python object whose type is current.
  ///
  /// \return The python object.
  const py::object &GetPythonObject() const { return obj_; }

  /// \brief Get the string representation of this object.
  ///
  /// \return The string representation.
  virtual std::string ToString() const { return (abs_ == nullptr) ? GetName() : abs_->ToString(); }

 private:
  /// \brief The AbstractBase of this AbstractType.
  abstract::AbstractBasePtr abs_;
  /// \brief The python object whose type is current.
  py::object obj_;
};

using AbstractTypePtr = std::shared_ptr<AbstractType>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_ABSTRACT_TYPE_H_
