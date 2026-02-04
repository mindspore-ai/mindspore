/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

// NOTICE: This header file should only be included once in the whole project.
// We change the cpp file to header file, to avoid MSVC compiler problem.
#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_PY_EXECUTE_PY_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_PY_EXECUTE_PY_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "ir/tensor_new.h"
#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

#include "include/utils/pipeline/fallback.h"
#include "mindspore/ops/infer/py_execute.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/python_utils.h"
#include "include/utils/tensor_py.h"
#include "include/utils/python_adapter.h"
#include "include/utils/pipeline/python_fallback_running.h"
#include "mindspore/ccsrc/include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/frontend/jit/ps/parse/data_converter.h"
#include "mindspore/ccsrc/frontend/jit/ps/parse/resolve.h"

namespace py = pybind11;
namespace mindspore {
namespace abstract {
using PyObjectWrapperPtr = std::shared_ptr<parse::PyObjectWrapper>;
namespace pyexecute_user_data_catcher {
std::pair<bool, ValuePtr> PyExecuteUserDataCatcher(const AbstractBasePtr &element_abs) {
  MS_EXCEPTION_IF_NULL(element_abs);
  if (element_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
    const auto &data = element_abs->user_data<kernel::PyExecuteOutputUserData>();
    MS_EXCEPTION_IF_NULL(data);
    auto python_obj = std::make_shared<parse::PyObjectWrapper>(data->obj, "graph python obj");
    return {true, python_obj};
  }
  return {false, nullptr};
}

struct PyExecuteUserDataCatcherRegister {
  PyExecuteUserDataCatcherRegister() noexcept {
    abstract::AbstractBase::set_pyexecute_user_data_catcher(
      [](const AbstractBasePtr &element_abs) { return PyExecuteUserDataCatcher(element_abs); });
  }
  ~PyExecuteUserDataCatcherRegister() {}
} pyexecute_user_data_catcher_register;
}  // namespace pyexecute_user_data_catcher
}  // namespace abstract

class PyExecuteInitializer {
 public:
  PyExecuteInitializer() {
    mindspore::ops::PyExecuteInfer::set_infer_handler(CppInferShapeAndTypePy);
    mindspore::opt::set_launch_handler(CppInferShapeAndTypePy);
  }

  ~PyExecuteInitializer() = default;

 private:
  static ValuePtr GetValueByAbstract(const abstract::AbstractBase *abstract);

  static ValuePtr ConstructEmptyTupleValue(const ValuePtr &structural);

  static std::pair<ValuePtr, size_t> ConstructInputValue(const ValuePtr &value,
                                                         const std::vector<abstract::AbstractBase *> &input_abstract,
                                                         size_t input_index);

  static ValuePtr ConstructInputValues(const PrimitivePtr &prim,
                                       const std::vector<abstract::AbstractBase *> &input_abstract);

  static abstract::AbstractBasePtr GenerateAbstract(const py::object &output);

  static abstract::AbstractBasePtr PyExecuteInferPy(const PrimitivePtr &primitive, const ValuePtr &input_value);

  static abstract::AbstractBasePtr CppInferShapeAndTypePy(const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBase *> &args_abs_list);
};

static PyExecuteInitializer py_execute_initializer;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_PY_EXECUTE_PY_H_
