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
#ifndef MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_
#define MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_

#include <vector>
#include <string>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

#include "include/common/fallback.h"
#include "mindspore/core/ops/py_execute.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils_py.h"
#include "mindspore/ccsrc/include/common/utils/python_adapter.h"
#include "mindspore/ccsrc/include/common/utils/python_fallback_running.h"
#include "mindspore/ccsrc/pipeline/jit/parse/data_converter.h"
#include "mindspore/ccsrc/pybind_api/ir/tensor_py.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "mindspore/ccsrc/backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

namespace py = pybind11;
namespace mindspore {
static py::object CallPythonGetGlobalParams() {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_get_dict = "get_global_params";
  return python_adapter::CallPyModFn(mod, python_get_dict);
}

class PyExecuteInitializer {
 public:
  PyExecuteInitializer() {
    mindspore::ops::PyExecuteInfer::set_infer_handler(PyExecuteInferPy);
    mindspore::opt::dynamic_shape::set_cpp_infer_py_handler(CppInferShapeAndTypePy);
  }

  ~PyExecuteInitializer() = default;

 private:
  static abstract::AbstractBasePtr PyExecuteInferPy(const std::vector<AbstractBasePtr> &input_args) {
    const auto &script_abs = input_args[0];
    const auto &script = script_abs->BuildValue();
    const auto &script_str = dyn_cast<StringImm>(script);

    const auto &keys_tuple_abs = input_args[1];
    const auto &keys_tuple = keys_tuple_abs->BuildValue();
    const auto &keys = dyn_cast<ValueSequence>(keys_tuple);

    // Process PyExecute("None", (), (), io)
    // Since the backend converts the empty tuple into an empty tensor(not keep ValueSequence),
    // so special handling of None is required.
    if (script->ToString() == "None") {
      const auto &output = py::none();
      MS_LOG(DEBUG) << "Python output type: " << py::str(output.get_type()) << ", output: " << output;
      PushPyExecuteOutput(output);
      const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
      return abstract::MakeAbstract(infer_shape, kFloat64);
    }

    if (keys == nullptr) {
      MS_LOG(DEBUG) << "The keys is not tuple value, but got " << keys_tuple->ToString();
      const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
      return abstract::MakeAbstract(infer_shape, kFloat64);
    }
    constexpr auto number_two = 2;
    const auto &values_tuple_abs = input_args[number_two];
    const auto &values_tuple = values_tuple_abs->BuildValue();
    if (values_tuple == kAnyValue) {
      MS_LOG(EXCEPTION) << "Value tuple should not be anyvalue.";
    }
    const auto &values = dyn_cast<ValueSequence>(values_tuple);
    if (values == nullptr) {
      MS_LOG(DEBUG) << "The values is not tuple value, but got " << keys_tuple->ToString();
      const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
      return abstract::MakeAbstract(infer_shape, kFloat64);
    }
    MS_LOG(DEBUG) << "script: " << script->ToString() << ", keys_tuple: " << keys_tuple->ToString()
                  << ", values_tuple: " << values_tuple->ToString();

    py::gil_scoped_acquire gil_acquire;
    py::dict local_dict;
    for (size_t i = 0; i < keys->size(); ++i) {
      const auto &key = (*keys)[i];
      const auto &key_str = dyn_cast<StringImm>(key);
      MS_EXCEPTION_IF_NULL(key_str);
      const auto &value = (*values)[i];
      const auto &tuple_abs = values_tuple_abs->cast<abstract::AbstractSequencePtr>();
      const auto &value_abs = (*tuple_abs)[i];
      if (value->isa<tensor::Tensor>()) {
        if (value_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
          const auto &output_data = value_abs->user_data<kernel::PyExecuteOutputUserData>();
          auto obj = output_data->obj;
          MS_LOG(DEBUG) << "input[" << i << "], obj: " << obj;
          local_dict[py::str(key_str->value())] = obj;
        } else {
          const auto &py_tensor = ValueToPyData(value);
          MS_LOG(DEBUG) << "input[" << i << "], py_tensor: " << py_tensor;
          local_dict[py::str(key_str->value())] = py_tensor;
        }
        continue;
      } else if (value->isa<StringImm>()) {
        const auto &str_imm = value->cast<StringImmPtr>();
        const auto &py_str = py::str(str_imm->value());
        MS_LOG(DEBUG) << "input[" << i << "], py_str: " << py_str;
        local_dict[py::str(key_str->value())] = py_str;
        continue;
      }
      MS_LOG(DEBUG) << "input[" << i << "], value: " << value;
      local_dict[py::str(key_str->value())] = value;
    }
    const auto &global_dict = CallPythonGetGlobalParams();
    const auto &py_script = py::str(script_str->value());
    auto params = py::tuple(number_two);
    params[0] = global_dict;
    params[1] = local_dict;
    MS_LOG(DEBUG) << "Python script: " << py_script << ", local_dict: " << local_dict;
    try {
      mindspore::ScopedFallbackRunning fallback_running;
      const auto &output = parse::data_converter::CallPythonScript(py_script, params);
      MS_LOG(DEBUG) << "Python output type: " << py::str(output.get_type()) << ", output: " << output;
      PushPyExecuteOutput(output);
      if (py::isinstance<tensor::Tensor>(output)) {
        const auto &tensor = output.cast<tensor::TensorPtr>();
        const auto &infer_shape = std::make_shared<abstract::Shape>(tensor->shape());
        return abstract::MakeAbstract(infer_shape, tensor->Dtype());
      }
    } catch (const py::error_already_set &e) {
      auto error_type_name = py::cast<std::string>(python_adapter::GetPyObjAttr(e.type(), "__name__"));
      auto error_iter = exception_types_map.find(error_type_name);
      if (error_iter != exception_types_map.end()) {
        auto &handler = LogWriter::GetExceptionHandler();
        if (handler != nullptr) {
          handler(error_iter->second, py::str(e.value()));
        }
      }
      throw std::runtime_error(py::str(e.value()));
    }

    const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
    return abstract::MakeAbstract(infer_shape, kFloat64);
  }

  static abstract::AbstractBasePtr CppInferShapeAndTypePy(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                                          const AbstractBasePtrList &args_spec_list) {
    // We can't catch the pybind11 exception by py::builtin_exception or its base class,
    // so we have to list all pybind11 exceptions and catch one by one here.
    try {
      const auto &abs = opt::CppInferShapeAndType(primitive, args_spec_list);
      MS_LOG(DEBUG) << "The abstract of " << cnode->fullname_with_scope() << " changes from " << cnode->abstract()
                    << " to " << abs;
      return abs;
    } catch (const py::type_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::type_error(ss.str());
    } catch (const py::value_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::value_error(ss.str());
    } catch (const py::index_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::index_error(ss.str());
    } catch (const py::key_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::key_error(ss.str());
    } catch (const py::attribute_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::attribute_error(ss.str());
    } catch (const py::name_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::name_error(ss.str());
    } catch (const py::assertion_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::assertion_error(ss.str());
    } catch (const py::base_exception &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::base_exception(ss.str());
    } catch (const py::keyboard_interrupt &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::keyboard_interrupt(ss.str());
    } catch (const py::stop_iteration &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::stop_iteration(ss.str());
    } catch (const py::overflow_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::overflow_error(ss.str());
    } catch (const py::zero_division_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::zero_division_error(ss.str());
    } catch (const py::environment_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::environment_error(ss.str());
    } catch (const py::io_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::io_error(ss.str());
    } catch (const py::os_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::os_error(ss.str());
    } catch (const py::memory_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::memory_error(ss.str());
    } catch (const py::unbound_local_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::unbound_local_error(ss.str());
    } catch (const py::not_implemented_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::not_implemented_error(ss.str());
    } catch (const py::indentation_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::indentation_error(ss.str());
    } catch (const py::runtime_warning &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw py::runtime_warning(ss.str());
    } catch (const std::runtime_error &e) {
      std::stringstream ss;
      ss << e.what() << ".\n\n" << trace::GetDebugInfo(cnode->debug_info());
      throw std::runtime_error(ss.str());
    }
  }
};

static PyExecuteInitializer py_execute_initializer;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_
