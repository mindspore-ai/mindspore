/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_UTILS_H
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_UTILS_H

#include "ir/anf.h"

namespace mindspore {
namespace expander {
/// \brief OpPrimPyRegister defines the singleton to save primitivepy which has no attrs.
class OpPrimPyRegister {
 public:
  /// \brief Destructor of OpPrimPyRegister.
  ~OpPrimPyRegister() {}

  /// \brief Get the OpPrimPyRegister singleton.
  ///
  /// \return The OpPrimPyRegister singleton.
  static OpPrimPyRegister &GetInstance() {
    static OpPrimPyRegister instance{};
    return instance;
  }

  /// \brief Get PrimPyMap of the OpPrimPyRegister singleton.
  ///
  /// \return The PrimPyMap of the OpPrimPyRegister singleton.
  const HashMap<std::string, ValuePtr> &GetPrimPyMap() const { return primpy_map_; }

  /// \brief Add an element into the PrimPyMap of the OpPrimPyRegister singleton.
  ///
  /// param[in] name The operator name.
  /// param[in] primpy The primitivepy of the operator.
  void SetPrimPyMap(const std::string &name, const ValuePtr &primpy) { primpy_map_[name] = primpy; }

  /// \brief Clear the PrimPyMap before the pyobject destroyed.
  void Clear() { primpy_map_.clear(); }

 private:
  OpPrimPyRegister() {}
  HashMap<std::string, ValuePtr> primpy_map_;  // op_name, primpy
};

bool ConvertPrimToPrimPy(const FuncGraphPtr &graph);
ValuePtr ConvertPrimToPrimPy(const PrimitivePtr &primc);
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_UTILS_H
