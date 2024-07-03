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
#include "pipeline/jit/pi/graph_capture/abstract_wrapper.h"
#include <vector>
#include <utility>
#include <memory>

#include "include/common/utils/python_adapter.h"
#include "include/common/utils/convert_utils_py.h"
#include "abstract/abstract_function.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/pi/pi_jit_config.h"

namespace mindspore {
constexpr auto kAdapterFlag = "adapter_flag";
constexpr auto kTensorModule = "mindspore.common";
constexpr auto kInnerOpsModule = "mindspore.ops.operations._inner_ops";
using PyTensorConverter = std::function<py::object(const py::object &)>;

namespace {
py::object ConvertCppTensorToPyTensor(const py::object &cpp_tensor) {
  if (cpp_tensor.ptr() == nullptr || !py::isinstance<tensor::Tensor>(cpp_tensor)) {
    return py::object();
  }
  bool is_adapter_tensor =
    py::hasattr(cpp_tensor, kAdapterFlag) && py::cast<bool>(py::getattr(cpp_tensor, kAdapterFlag));
  py::module mod = python_adapter::GetPyModule(kTensorModule);
  auto py_tensor = python_adapter::CallPyModFn(mod, "Tensor", cpp_tensor, py::none(), py::none(), py::none(), true);
  if (is_adapter_tensor) {
    mod = python_adapter::GetPyModule(kInnerOpsModule);
    py_tensor = python_adapter::CallPyModFn(mod, "convert_to_adapter_tensor", py_tensor);
  }
  return py_tensor;
}

py::object ConvertToPyTensorOrParameter(const py::object &cpp_tensor) {
  if (cpp_tensor.ptr() == nullptr || !py::isinstance<tensor::Tensor>(cpp_tensor)) {
    return py::object();
  }
  // TODO(LiangZhibo): guard for parameter need to change later.
  auto tensor = py::cast<tensor::TensorPtr>(cpp_tensor);
  if (tensor->is_parameter()) {
    return cpp_tensor;
  }

  return ConvertCppTensorToPyTensor(cpp_tensor);
}

ValuePtr MaybeMakeEmptyTensor(const AbstractBasePtr &abs) {
  auto build_value = abs->BuildValue();
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> value_vec;
    for (auto &elem : abs_seq->elements()) {
      (void)value_vec.emplace_back(MaybeMakeEmptyTensor(elem));
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return std::make_shared<ValueTuple>(value_vec);
    } else {
      return std::make_shared<ValueList>(value_vec);
    }
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    const auto &elements = abs_dict->elements();
    std::vector<std::pair<ValuePtr, ValuePtr>> val_dict;
    for (auto &element : elements) {
      auto key_value = MaybeMakeEmptyTensor(element.first);
      auto val_value = MaybeMakeEmptyTensor(element.second);
      (void)val_dict.emplace_back(std::pair<ValuePtr, ValuePtr>{key_value, val_value});
    }
    return std::make_shared<ValueDictionary>(val_dict);
  }
  if (build_value == kValueAny && abs->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = abs->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = abs_tensor->element()->BuildType();
    ShapeVector tensor_shape = abs_tensor->shape()->shape();
    auto tensor = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    if (abs->isa<abstract::AbstractRefTensor>()) {
      auto abs_ref_tensor = abs->cast<abstract::AbstractRefPtr>();
      // We only need the parameter name, it was used to find the python Parameter object later
      auto param_info = std::make_shared<ParamInfo>();
      param_info->set_name(abs_ref_tensor->ref_key_value()->ToString());
      tensor->set_param_info(param_info);
    }
    return tensor;
  }
  return build_value;
}

py::object ConvertToPythonTensor(const py::object &obj) {
  constexpr auto ms_class_attr = "__ms_class__";
  if (py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr))) {
    return obj;
  }
  if (py::isinstance<tensor::Tensor>(obj)) {
    return ConvertToPyTensorOrParameter(obj);
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    auto obj_tuple = py::cast<py::tuple>(obj);
    py::tuple ret(obj_tuple.size());
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      ret[i] = ConvertToPythonTensor(obj_tuple[i]);
    }
    if (py::isinstance<py::list>(obj)) {
      return ret.cast<py::list>();
    }
    return ret;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto obj_dict = py::cast<py::dict>(obj);
    for (auto item : obj_dict) {
      obj_dict[item.first] = ConvertToPythonTensor(py::cast<py::object>(item.second));
    }
    return obj_dict;
  }
  return obj;
}

py::object ConvertToPyObj(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractNone>()) {
    return py::none();
  }

  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func->func_graph();
    if (fg != nullptr) {
      auto obj = fg->python_obj();
      if (obj != nullptr && obj->isa<parse::PyObjectWrapper>()) {
        return obj->cast_ptr<parse::PyObjectWrapper>()->obj();
      }
    }
    return py::object();
  }
  if (abs->isa<abstract::MetaFuncGraphAbstractClosure>()) {
    auto meta_fg = abs->cast<abstract::MetaFuncGraphAbstractClosurePtr>()->meta_func_graph();
    if (meta_fg != nullptr) {
      return py::cast(meta_fg);
    }
    return py::object();
  }
  auto build_value = MaybeMakeEmptyTensor(abs);
  auto py_obj = ValueToPyData(build_value, abs);
  // Return none means failed converting.
  if (py::isinstance<py::none>(py_obj)) {
    return py::object();
  }

  // TODO(LiangZhibo): One stage and two stage should do same process, fix later.
  if (pijit::kPIJitConfigDefault.GetBoolConfig(pijit::GraphJitConfig::kTraceFlag)) {
    return ConvertToPythonTensor(py_obj);
  }

  return py_obj;
}
}  // namespace

py::object AbstractWrapper::ConvertToPyObject(const AbstractWrapperPtr &wrapper) {
  if (wrapper == nullptr || wrapper->abstract() == nullptr) {
    return py::object();
  }
  return ConvertToPyObj(wrapper->abstract());
}

py::object AbstractWrapper::ConvertToPyObject(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return py::object();
  }
  return ConvertToPyObj(abstract);
}

std::string AbstractWrapper::ToString() const {
  if (abstract_ == nullptr) {
    return "AbstractWrapper: NULL";
  }
  return "AbstractWrapper: " + abstract_->ToString();
}
}  // namespace mindspore
