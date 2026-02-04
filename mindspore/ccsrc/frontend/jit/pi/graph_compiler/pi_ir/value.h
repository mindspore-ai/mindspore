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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_VALUE_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_VALUE_H_

#include <memory>
#include <string>
#include "frontend/jit/pi/graph_compiler/pi_ir/node.h"
#include "frontend/jit/pi/graph_compiler/pi_ir/scope.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace ir {
namespace py = pybind11;

/// \brief Value is is the class which represent a python object
class Value : public Node {
 public:
  /**
   * \brief The constructor of value node.
   *
   * \param[in] value the python object.
   *
   * \return The instance of value node.
   */
  explicit Value(const py::object &value) : Value(value, "", kScopeLocal) {}

  /**
   * \brief The constructor of value node.
   *
   * \param[in] value the python object.
   * \param[in] scope the located of python object.
   *
   * \return The instance of value node.
   */
  Value(const py::object &value, Scope scope) : Value(value, "", scope) {}

  /**
   * \brief The constructor of value node.
   *
   * \param[in] value the python object.
   * \param[in] name the name of python object.
   *
   * \return The instance of value node.
   */
  Value(const py::object &value, const std::string &name) : Value(value, name, kScopeLocal) {}

  /**
   * \brief The constructor of value node.
   *
   * \param[in] value the python object.
   * \param[in] name the name of python object.
   * \param[in] scope the located of python object.
   *
   * \return The instance of value node.
   */
  Value(const py::object &value, const std::string &name, Scope scope) : value_(value), name_(name), scope_(scope) {}

  // \brief Destructor.
  ~Value() override = default;
  JIT_DECLARE_PARENT(Value, Node);

  bool operator==(const Value &other) { return value_.ptr() == other.value_.ptr(); }

  /**
   * \brief Get python object.
   *
   * \return the python object.
   */
  const py::object &GetValue() const { return value_; }

  /**
   * \brief Set the python object.
   *
   * \param[in] value the python object.
   */
  void SetValue(const py::object &value) { value_ = value; }

  /**
   * \brief Get the name of python object.
   *
   * \return the name of python object.
   */
  const std::string &GetName() const { return name_; }

  /**
   * \brief Set the name of python object.
   *
   * \param[in] name the name of python object.
   */
  void SetName(const std::string &name) { name_ = name; }

  /**
   * \brief Get the scope of the value.
   *
   * \return the scope of the value.
   */
  Scope GetScope() const { return scope_; }

  /**
   * \brief Get the description of this value.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The python object.
  py::object value_;
  /// \brief The name of python object.
  std::string name_;
  /// Where the value is located
  Scope scope_;
};

using ValuePtr = std::shared_ptr<Value>;
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_VALUE_H_
