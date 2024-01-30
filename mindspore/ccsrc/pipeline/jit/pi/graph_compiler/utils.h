/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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
#ifndef MINDSPORE_PI_JIT_COMPILER_UTILS_H_
#define MINDSPORE_PI_JIT_COMPILER_UTILS_H_

#include <string>
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace pijit {
class GraphUtils {
 public:
  // object is const when it has attr(const_arg) and the value of attr(const_arg) is true
  static bool IsConst(const py::object &obj) { return IsAttrEnabled(obj, "const_arg"); }

  // object is mutable when it has attr(__ms_mutable__) and the value of attr(__ms_mutable__) is true
  static bool IsMutable(const py::object &obj) { return IsAttrEnabled(obj, "__ms_mutable__"); }

  // object is dynamic length when it has attr(__ms_dynamic_len__) and the value of attr(__ms_dynamic_len__) is true
  static bool IsDynamicLength(const py::object &obj) { return IsAttrEnabled(obj, "__ms_dynamic_len__"); }

  // object is enable_tuple_broaden when it has attr(enable_tuple_broaden) and the value of attr(enable_tuple_broaden)
  // is true
  static bool IsTupleBroadenEnable(const py::object &obj) { return IsAttrEnabled(obj, "enable_tuple_broaden"); }

  // object is has init when it has attr(has_init) and the value of attr(has_init) is true
  static bool HasInit(const py::object &obj) { return IsAttrEnabled(obj, "has_init"); }

  static bool IsTupleCanBroaden(const py::object &obj);

  static bool IsGradForScalar(const py::object &obj);

  static bool IsTensor(const py::object &obj);

  static AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden = true);

  static PrimitivePtr GetPrimitive(int op_code);

  static AnfNodePtr GetPrimOrMetaFuncGraph(int op_code);

  static std::string OpCompareArgToGraphName(int oparg);

  static std::string OpCodeToGraphName(int op_code);

  static AnfNodePtr GetMetaFuncGraph(int op_code);

  static AnfNodePtr GetMetaFuncGraph(const std::string &name);

  static AnfNodePtr ConvertPythonObjectToAnfNode(const py::object &object);

 private:
  // object has the attr and the value of attr is true
  static bool IsAttrEnabled(const py::object &obj, const std::string &attr) {
    return py::hasattr(obj, common::SafeCStr(attr)) && py::cast<bool>(py::getattr(obj, common::SafeCStr(attr)));
  }
};
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_COMPILER_UTILS_H_
