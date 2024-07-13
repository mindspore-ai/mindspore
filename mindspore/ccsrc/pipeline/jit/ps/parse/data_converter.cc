/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/parse/data_converter.h"
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/structure_ops.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "ir/func_graph_cloner.h"
#include "ir/cell.h"
#include "ir/dtype.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "include/common/fallback.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/primfunc_utils.h"
#include "frontend/operator/composite/multitype_funcgraph.h"

namespace mindspore {
namespace parse {
namespace {
struct PyDataToValueRegister {
  PyDataToValueRegister() noexcept {
    python_adapter::PyAdapterCallback::SetPyDataToValueHandler(data_converter::PyDataToValue);
  }
} callback_register;
}  // namespace
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaTensorPtr = mindspore::tensor::MetaTensorPtr;
using CSRTensor = mindspore::tensor::CSRTensor;
using CSRTensorPtr = mindspore::tensor::CSRTensorPtr;
using COOTensor = mindspore::tensor::COOTensor;
using COOTensorPtr = mindspore::tensor::COOTensorPtr;
using MapTensor = mindspore::tensor::MapTensor;
using MapTensorPtr = mindspore::tensor::MapTensorPtr;

using InstanceCheckFunc = std::function<bool(const py::object &)>;
using InstanceConvertFunc = std::function<ValuePtr(const py::object &, bool, const TypePtr &, const ValuePtrList &)>;
static constexpr int kBit8 = 8;
static constexpr int kBit16 = 16;
static constexpr int kBit32 = 32;
static constexpr int kBit64 = 64;

class DataConvertFunc {
 public:
  explicit DataConvertFunc(InstanceConvertFunc convert_func) : convert_func_(std::move(convert_func)) {}

  virtual ~DataConvertFunc() = default;

  virtual bool Matched(const py::object &obj) = 0;

  ValuePtr ConvertPyObject(const py::object &obj, bool use_sig, const TypePtr &dtype,
                           const ValuePtrList &args_value_list = {}) {
    if (convert_func_ == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "convert func is null";
    }
    return convert_func_(obj, use_sig, dtype, args_value_list);
  }

 private:
  InstanceConvertFunc convert_func_ = nullptr;
};

using DataConvertFuncPtr = std::shared_ptr<DataConvertFunc>;

using ArgsObjConvertFunc = std::function<ValuePtr(const py::object &)>;
using ArgsObjSigConvertFunc = std::function<ValuePtr(const py::object &, bool)>;
using ArgsObjTypeConvertFunc = std::function<ValuePtr(const py::object &, const TypePtr &)>;
using ArgsObjArgsValueConvertFunc = std::function<ValuePtr(const py::object &, const ValuePtrList &)>;

// Convert the data according to instance type
template <typename T>
class ByTypeDataConvertFunc : public DataConvertFunc {
 public:
  explicit ByTypeDataConvertFunc(const InstanceConvertFunc &convert_func)
      : DataConvertFunc(convert_func), check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ValuePtr &converted_type)
      : DataConvertFunc([converted_type](const py::object &, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return converted_type;
        }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjSigConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjTypeConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &dtype,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, dtype); }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjArgsValueConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &,
                                       const ValuePtrList &args_value_list) -> ValuePtr {
          return convert_func(obj, args_value_list);
        }),
        check_func_(py::isinstance<T>) {}

  ~ByTypeDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return check_func_ != nullptr ? check_func_(obj) : false; }

 private:
  InstanceCheckFunc check_func_ = nullptr;
};

// Convert the data according to object attribute.
class ByAttrDataConvertFunc : public DataConvertFunc {
 public:
  ByAttrDataConvertFunc(const ArgsObjConvertFunc &convert_func, const std::string &attr_name,
                        const std::string &cell_list_from_top = "")
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        attr_name_(attr_name),
        cell_list_from_top_(cell_list_from_top) {}

  ByAttrDataConvertFunc(const ArgsObjSigConvertFunc &convert_func, const std::string &attr_name,
                        const std::string &cell_list_from_top = "")
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        attr_name_(attr_name),
        cell_list_from_top_(cell_list_from_top) {}

  ~ByAttrDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override {
    return py::hasattr(obj, attr_name_.c_str()) && !py::hasattr(obj, cell_list_from_top_.c_str());
  }

 private:
  std::string attr_name_;
  std::string cell_list_from_top_;
};

// Convert the data according to match function.
class ByFuncDataConvertFunc : public DataConvertFunc {
 public:
  ByFuncDataConvertFunc(const InstanceCheckFunc &match_func, const ArgsObjConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        match_func_(match_func) {}

  ByFuncDataConvertFunc(const InstanceCheckFunc &match_func, const ArgsObjSigConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        match_func_(match_func) {}

  ~ByFuncDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return match_func_ != nullptr ? match_func_(obj) : false; }

 private:
  InstanceCheckFunc match_func_ = nullptr;
};

FuncGraphPtr ConvertToBpropCut(const py::object &obj) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_key = results[0];
  py::function bprop_func = py::getattr(obj, CUSTOM_BPROP_NAME);

  auto bprop_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto fake_bprop = std::make_shared<PrimitivePy>("bprop_cut");
  fake_bprop->AddBackwardHookFn(0, bprop_func);
  (void)fake_bprop->AddAttr(CUSTOM_BPROP_NAME, MakeValue(true));
  outputs.push_back(NewValueNode(fake_bprop));

  py::object code_obj = py::getattr(bprop_func, "__code__");
  // Three parameters self, out and dout need to be excluded
  constexpr auto kBpropExcludeParamNum = 3;
  size_t inputs_num = py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - kBpropExcludeParamNum;
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = bprop_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = bprop_graph->add_parameter();
  auto p2 = bprop_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  bprop_graph->set_output(bprop_graph->NewCNode(std::move(outputs)));
  data_converter::SetObjGraphValue(obj_key, bprop_graph);
  return bprop_graph;
}

namespace {
ValuePtr ConvertTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  auto tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(tuple[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  auto res = std::make_shared<ValueTuple>(value_list);
  return res;
}

bool IsNamedTuple(const py::object &obj) { return py::hasattr(obj, "_fields") && py::isinstance<py::tuple>(obj); }

ValuePtr ConvertNamedTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python NamedTuple";
  if (!py::hasattr(obj, "_asdict")) {
    return nullptr;
  }
  auto asdict_fn = obj.attr("_asdict");
  auto asdict_obj = asdict_fn();
  auto dict_values = asdict_obj.cast<py::dict>();
  std::vector<ValuePtr> keys;
  std::vector<ValuePtr> values;
  for (auto item : dict_values) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = ConvertData(py::cast<py::object>(item.first), &key, use_signature) &&
                   ConvertData(py::cast<py::object>(item.second), &value, use_signature);
    if (!success) {
      return nullptr;
    }
    MS_LOG(DEBUG) << key->ToString() << ", " << value->ToString();
    keys.push_back(key);
    values.push_back(value);
  }
  auto obj_name = obj.attr("__class__").attr("__name__");
  std::string sub_class_name = py::str(obj_name).cast<std::string>();
  return std::make_shared<ValueNamedTuple>(sub_class_name, keys, values);
}

ValuePtr ConvertStubTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  auto tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertStubData(tuple[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";
  PyRecursionScope scope(obj);

  auto list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  auto res = std::make_shared<ValueList>(value_list);
  return res;
}

ValuePtr ConvertStubList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";
  PyRecursionScope scope(obj);

  auto list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertStubData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueList>(value_list);
}

ValuePtr ConvertCellList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting cell list";
  PyRecursionScope scope(obj);

  py::sequence list = obj;
  std::vector<ValuePtr> value_list;

  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  bool is_celllist = py::cast<bool>(python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_CELL_LIST, obj));
  for (const auto &element : list) {
    // An element will directly convert to InterpretedObject if:
    //   1. The container is not a cell list object.
    //   2. The element should be single cell (cell with no __cell_as_list__ attr).
    bool to_interpret = !is_celllist && py::isinstance<Cell>(element);
    if (to_interpret) {
      value_list.push_back(std::make_shared<parse::InterpretedObject>(element));
      continue;
    }
    ValuePtr out = nullptr;
    bool success = ConvertData(element, &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertDict(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python dict";
  PyRecursionScope scope(obj);

  auto dict_values = obj.cast<py::dict>();
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  for (auto item : dict_values) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = ConvertData(py::cast<py::object>(item.first), &key, use_signature) &&
                   ConvertData(py::cast<py::object>(item.second), &value, use_signature);
    if (!success) {
      return nullptr;
    }
    (void)key_values.emplace_back(key, value);
  }
  auto res = std::make_shared<ValueDictionary>(key_values);
  res->set_user_data<py::object>("origin_object", std::make_shared<py::object>(obj));
  return res;
}

ValuePtr ConvertModuleNameSpace(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting python module";
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object module_namespace = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MODULE_NAMESPACE, obj);
  auto converted = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_MODULE, module_namespace, obj);
  MS_LOG(DEBUG) << "name_space: " << converted->ToString();
  return converted;
}

ValuePtr ConvertMsClass(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting ms class";
  // Convert class instance decorated with jit_class.
  if (py::hasattr(obj, PYTHON_PARSE_METHOD)) {
    MS_LOG(DEBUG) << "Convert obj to func graph.";
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
    func_graph->set_python_obj(python_obj);
    return func_graph;
  }
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object name = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MS_CLASS_NAME, obj);
  auto cls_name = py::cast<std::string>(name);
  return std::make_shared<MsClassObject>(obj, cls_name);
}

ValuePtr ConvertPrimitiveClassType(const py::object &obj) {
  // need check the primitive is class type or instance
  auto obj_type = data_converter::GetObjType(obj);
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  return nullptr;
}

ValuePtr ConvertPrimitive(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting primitive object " << use_signature;

  auto class_type = ConvertPrimitiveClassType(obj);
  if (class_type != nullptr) {
    return class_type;
  }
  py::object adapter_obj = obj;
  if (py::hasattr(obj, "__setattr_flag__")) {
    if (py::hasattr(obj, "_clone")) {
      auto clone_fn = obj.attr("_clone");
      adapter_obj = clone_fn();
    }
  }
  auto prim_adapter = adapter_obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(prim_adapter);
  auto primitive = prim_adapter->attached_primitive();
  if (primitive == nullptr) {
    primitive = std::make_shared<PrimitivePy>(adapter_obj);
    prim_adapter->set_attached_primitive(primitive);
  }

  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(primitive->name(), primitive);
  }
  return primitive;
}

ValuePtr ConvertPrimitiveFunction(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting primitive function";
  auto class_type = ConvertPrimitiveClassType(obj);
  if (class_type != nullptr) {
    return class_type;
  }
  auto prim_func_adapter = obj.cast<PrimitiveFunctionAdapterPtr>();
  MS_EXCEPTION_IF_NULL(prim_func_adapter);
  auto cpp_primitive_func = prim_func_adapter->attached_primitive_function();
  if (cpp_primitive_func == nullptr) {
    auto prim_name = py::getattr(obj, "name").cast<std::string>();
    return std::make_shared<prim::DoTransPrimitiveFunction>(std::make_shared<Primitive>(prim_name));
  }
  return cpp_primitive_func;
}

ValuePtr ConvertMetaFuncGraph(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting MetaFuncGraph object";
  auto meta = obj.cast<MetaFuncGraphPtr>();
  if (meta == nullptr) {
    MS_LOG(ERROR) << "Resolve MetaFuncGraph error, get ptr is null";
    return nullptr;
  }
  auto multi = meta->cast<prim::MultitypeFuncGraphPtr>();
  if (multi != nullptr) {
    multi->set_meta_obj(obj);
  }
  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta);
  }
  return meta;
}

ValuePtr ConvertFuncGraph(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting FuncGraph object";
  auto func_graph = obj.cast<FuncGraphPtr>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Resolve FuncGraph error, get ptr is null";
    return nullptr;
  }
  func_graph->set_attr("is_load", MakeValue(true));
  return func_graph;
}

ValuePtr ConvertSlice(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting slice object";

  auto convert_func = [obj](const std::string &attr) -> ValuePtr {
    auto py_attr = py::getattr(obj, attr.c_str());
    if (py::isinstance<py::none>(py_attr)) {
      return kNone;
    }
    if (py::isinstance<py::int_>(py_attr)) {
      auto value = py::cast<int64_t>(py_attr);
      return MakeValue(value);
    }
    if (py::isinstance<Tensor>(py_attr)) {
      return py::cast<TensorPtr>(py_attr);
    }
    if (IsStubTensor(py_attr)) {
      return ConvertStubTensor(py_attr);
    }
    MS_LOG(EXCEPTION) << "Attribute '" << attr << "' of " << py::str(obj)
                      << " should be int or Tensor with Int type but got " << py::str(py_attr);
  };
  ValuePtr start = convert_func(kSliceStart);
  ValuePtr stop = convert_func(kSliceStop);
  ValuePtr step = convert_func(kSliceStep);
  return std::make_shared<ValueSlice>(start, stop, step);
}

ValuePtr ConvertCellObjToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list) {
  FuncGraphPtr func_graph = ConvertToFuncGraph(obj, args_value_list);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }
  // if the cell object has specified bprop, it has user-defined bprop function parse and record it
  if (py::hasattr(obj, CUSTOM_BPROP_NAME)) {
    bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
    FuncGraphPtr bprop_graph =
      enable_bprop_debug ? ConvertToBpropCut(obj) : ConvertToFuncGraph(obj, {}, PYTHON_MOD_GET_BPROP_METHOD);
    if (bprop_graph != nullptr) {
      (void)func_graph->transforms().emplace(CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
      (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(func_graph));
      func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      func_graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    }
  }
  if (py::hasattr(obj, STAGE_NAME)) {
    auto stage = py::cast<int>(py::getattr(obj, STAGE_NAME));
    func_graph->set_stage(stage);
  }
  if (py::hasattr(obj, SEGMENT_NAME)) {
    auto segment = py::cast<int>(py::getattr(obj, SEGMENT_NAME));
    func_graph->set_segment(segment);
  }
  auto cell = py::cast<CellPtr>(obj);
  if (cell != nullptr && cell->HasAttr(kAttrRandomOpSnapShot)) {
    auto value = cell->GetAttr(kAttrRandomOpSnapShot);
    MS_EXCEPTION_IF_NULL(value);
    func_graph->set_attr(kAttrRandomOpSnapShot, value);
  }
  return func_graph;
}

ValuePtr ConvertConstantNumpyNumber(const py::object &obj, ResolveType obj_type) {
  if (obj_type == RESOLVE_TYPE_NUMPY_INT_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy int64_t number:" << (std::string)py::str(obj);
    return MakeValue(py::cast<int64_t>(obj));
  }
  if (obj_type == RESOLVE_TYPE_NUMPY_FLOAT_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy float number::" << (std::string)py::str(obj);
    return MakeValue(py::cast<float>(obj));
  }
  if (obj_type == RESOLVE_TYPE_NUMPY_BOOL_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy bool_ number::" << (std::string)py::str(obj);
    return MakeValue(py::cast<bool>(obj));
  }

  MS_LOG(ERROR) << "Convert numpy number type is invalid, obj: " << py::str(obj);
  return nullptr;
}

void CheckJITForbiddenAPI(const py::object &obj) {
  auto module = python_adapter::GetPyModule(PYTHON_MOD_MODULE);
  py::object res = python_adapter::CallPyModFn(module, PYTHON_MOD_GET_MODULE_AND_NAME_INFO, obj);
  if (!py::isinstance<py::none>(res)) {
    auto obj_info = py::cast<py::list>(res);
    auto obj_module = py::cast<std::string>(obj_info[0]);
    auto obj_name = py::cast<std::string>(obj_info[1]);
    auto obj_type = py::cast<std::string>(obj_info[2]);
    std::ostringstream oss;
    oss << "Failed to compile in GRAPH_MODE because the " << obj_type << " '" << obj_module << "." << obj_name
        << "' is not supported in 'construct' or function with @jit decorator. "
        << "Try to use the " << obj_type << " '" << obj_module << "." << obj_name << "' externally "
        << "such as initialized in the method '__init__' before assigning"
        << ".\nFor more details, please refer to "
        << "https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html \n";
    // Check if the API is decoratored by @jit_forbidden_register.
    bool is_jit_forbidden_register = data_converter::IsJITForbiddenAPI(obj);
    if (is_jit_forbidden_register) {
      MS_LOG(EXCEPTION) << oss.str();
    }
    // Check if the API's module is in the JIT forbidden module set.
    bool is_jit_forbidden_module =
      py::cast<bool>(python_adapter::CallPyModFn(module, PYTHON_MOD_IS_JIT_FORBIDDEN_MODULE, obj_info[0]));
    if (is_jit_forbidden_module) {
      MS_LOG(EXCEPTION) << oss.str();
    }
  }
}

ValuePtr ConvertOtherObj(const py::object &obj, bool forbid_reuse = false) {
  auto obj_type = data_converter::GetObjType(obj);
  MS_LOG(DEBUG) << "Converting the object(" << ((std::string)py::str(obj)) << ") detail type: " << obj_type << " ";
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    // Check JIT forbidden API
    CheckJITForbiddenAPI(obj);
    MS_LOG(DEBUG) << "Resolve the class type, need create class instance.";
    std::string desc = py::str(obj);
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD ||
      (obj_type == RESOLVE_TYPE_CLASS_INSTANCE && py::hasattr(obj, PYTHON_PARSE_METHOD))) {
    if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD) {
      // Check JIT forbidden API
      CheckJITForbiddenAPI(obj);
      // Check if the function is from a third-party library.
      py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
      bool is_third_party_function =
        python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_FROM_THIRD_PARTY_LIBRARY, obj).cast<bool>();
      if (is_third_party_function) {
        MS_LOG(DEBUG) << "Converting the function from third-party library: " << py::str(obj);
        return std::make_shared<InterpretedObject>(obj);
      }
    }
    MS_LOG(DEBUG) << "Convert the obj to func graph, type is " << obj_type;
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj, {}, PYTHON_MOD_GET_PARSE_METHOD, forbid_reuse);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    return func_graph;
  }
  if (obj_type == RESOLVE_TYPE_CLASS_INSTANCE) {
    MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert class instance: " << py::str(obj);
  }
  // Start RESOLVE_TYPE_INVALID.
  if (obj_type == RESOLVE_TYPE_NUMPY_INT_NUMBER || obj_type == RESOLVE_TYPE_NUMPY_FLOAT_NUMBER ||
      obj_type == RESOLVE_TYPE_NUMPY_BOOL_NUMBER) {
    return ConvertConstantNumpyNumber(obj, obj_type);
  }
  auto res = std::make_shared<InterpretedObject>(obj);
  MS_EXCEPTION_IF_NULL(res);
  MS_LOG(DEBUG) << "Get interpreted object: " << res->ToString();
  return res;
}

template <typename T>
ValuePtr ConvertNumberWithType(const T &obj, const TypePtr &dtype) {
  ValuePtr data = nullptr;
  auto int_dypte = dyn_cast<Int>(dtype);
  if (int_dypte != nullptr) {
    switch (int_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<Int8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<Int16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<Int32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<Int64Imm>(obj);
        break;
      default:
        data = std::make_shared<Int64Imm>(obj);
    }
    return data;
  }

  auto uint_dypte = dyn_cast<UInt>(dtype);
  if (uint_dypte != nullptr) {
    switch (uint_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<UInt8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<UInt16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<UInt32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<UInt64Imm>(obj);
        break;
      default:
        data = std::make_shared<UInt32Imm>(obj);
    }
    return data;
  }

  auto float_dypte = dyn_cast<Float>(dtype);
  if (float_dypte != nullptr) {
    switch (float_dypte->nbits()) {
      case kBit32:
        data = std::make_shared<FP32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<FP64Imm>(obj);
        break;
      default:
        data = std::make_shared<FP32Imm>(obj);
    }
    return data;
  }
  return nullptr;
}

ValuePtr ConvertIntegerWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_int64 = py::cast<int64_t>(obj);
  // The mutable _Bool class inherits from int, because base class 'bool' is a marked final.
  if (py::hasattr(obj, "__ms_mutable_bool__")) {
    bool obj_bool = obj_int64 != 0;
    return std::make_shared<BoolImm>(obj_bool);
  }
  if (dtype == nullptr) {
    return std::make_shared<Int64Imm>(obj_int64);
  }
  return ConvertNumberWithType<int64_t>(obj_int64, dtype);
}

ValuePtr ConvertFloatWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_float32 = py::cast<pyfloat>(obj);
  if (dtype == nullptr) {
    auto obj_double = py::cast<double>(obj);
    auto ret = std::make_shared<FP32Imm>(obj_float32);
    ret->set_prim_value(obj_double);
    return ret;
  }
  return ConvertNumberWithType<pyfloat>(obj_float32, dtype);
}

ValuePtr ConvertNameSpace(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python NameSpace";
  auto res = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, obj);
  MS_LOG(DEBUG) << "name_space: " << res->ToString();
  return res;
}

template <typename T, typename U>
ValuePtr PyCast(const py::object &obj) {
  return std::make_shared<T>(py::cast<U>(obj));
}

template <typename T>
ValuePtr ObjCast(const py::object &obj) {
  return obj.cast<T>();
}

static const std::vector<DataConvertFuncPtr> &GetDataConvertFuncs() {
  // Convert data by python object type.
  static const std::vector<DataConvertFuncPtr> data_convert_funcs{
    // AdapterTensor needs to be processed before Tensor because it inherits from Tensor.
    std::make_shared<ByFuncDataConvertFunc>(IsStubTensor, ConvertStubTensor),
    std::make_shared<ByFuncDataConvertFunc>(IsNamedTuple, ConvertNamedTuple),
    std::make_shared<ByTypeDataConvertFunc<Tensor>>(ObjCast<TensorPtr>),
    std::make_shared<ByAttrDataConvertFunc>(ConvertMsClass, PYTHON_MS_CLASS),
    std::make_shared<ByTypeDataConvertFunc<BaseTensor>>(ObjCast<BaseTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<py::tuple>>(ConvertTuple),
    std::make_shared<ByTypeDataConvertFunc<py::list>>(ConvertList),
    std::make_shared<ByTypeDataConvertFunc<py::bool_>>(PyCast<BoolImm, bool>),
    std::make_shared<ByTypeDataConvertFunc<py::int_>>(ConvertIntegerWithType),
    std::make_shared<ByTypeDataConvertFunc<py::float_>>(ConvertFloatWithType),
    std::make_shared<ByTypeDataConvertFunc<py::str>>(PyCast<StringImm, string>),
    std::make_shared<ByTypeDataConvertFunc<py::none>>(kNone),
    std::make_shared<ByTypeDataConvertFunc<MetaTensor>>(ObjCast<MetaTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<CSRTensor>>(ObjCast<CSRTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<COOTensor>>(ObjCast<COOTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<MapTensor>>(ObjCast<MapTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<py::ellipsis>>(kEllipsis),
    std::make_shared<ByTypeDataConvertFunc<py::module>>(ConvertModuleNameSpace),
    std::make_shared<ByTypeDataConvertFunc<Type>>(ObjCast<TypePtr>),
    std::make_shared<ByTypeDataConvertFunc<UMonad>>(ObjCast<UMonadPtr>),
    std::make_shared<ByTypeDataConvertFunc<IOMonad>>(ObjCast<IOMonadPtr>),
    std::make_shared<ByAttrDataConvertFunc>(ConvertNameSpace, PYTHON_CLASS_MEMBER_NAMESPACE),
    std::make_shared<ByTypeDataConvertFunc<py::dict>>(ConvertDict),
    std::make_shared<ByAttrDataConvertFunc>(ConvertDict, PYTHON_CELL_AS_DICT),
    std::make_shared<ByTypeDataConvertFunc<py::slice>>(ConvertSlice),
    std::make_shared<ByAttrDataConvertFunc>(ConvertCellList, PYTHON_CELL_AS_LIST, PYTHON_CELL_LIST_FROM_TOP),
    std::make_shared<ByTypeDataConvertFunc<Cell>>(ConvertCellObjToFuncGraph),
    std::make_shared<ByAttrDataConvertFunc>(ConvertPrimitive, PYTHON_PRIMITIVE_FLAG),
    std::make_shared<ByAttrDataConvertFunc>(ConvertPrimitiveFunction, PYTHON_PRIMITIVE_FUNCTION_FLAG),
    std::make_shared<ByTypeDataConvertFunc<MetaFuncGraph>>(ConvertMetaFuncGraph),
    std::make_shared<ByTypeDataConvertFunc<FuncGraph>>(ConvertFuncGraph),
  };
  return data_convert_funcs;
}

static const std::vector<DataConvertFuncPtr> &GetStubDataConvertFuncs() {
  // Convert data by python object type.
  static const std::vector<DataConvertFuncPtr> data_convert_funcs{
    std::make_shared<ByFuncDataConvertFunc>([](const py::object &obj) -> bool { return IsStubTensor(obj); },
                                            PyStubNodeCast),
    std::make_shared<ByTypeDataConvertFunc<py::tuple>>(ConvertStubTuple),
    std::make_shared<ByTypeDataConvertFunc<py::list>>(ConvertStubList),
  };
  return data_convert_funcs;
}

void RemoveRecomputeScope(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);

  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &origin_scope_name = node->scope()->name();
    if (origin_scope_name.compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0) {
      auto remove_recompute_scope = origin_scope_name.substr(strlen(kAttrRecompute) + 1);
      node->set_scope(std::make_shared<Scope>(remove_recompute_scope));
    }
  }
}
}  // namespace

bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature, const TypePtr &dtype, bool forbid_reuse) {
  // Check parameter valid
  if (data == nullptr) {
    MS_LOG(ERROR) << "The value pointer should not be null.";
    return false;
  }
  ValuePtr converted = nullptr;
  bool matched = false;
  const auto &converters = GetDataConvertFuncs();
  for (auto &converter : converters) {
    if (converter->Matched(obj)) {
      converted = converter->ConvertPyObject(obj, use_signature, dtype);
      matched = true;
      break;
    }
  }
  if (!matched) {
    converted = ConvertOtherObj(obj, forbid_reuse);
  }
  *data = converted;
  return converted != nullptr;
}

bool ConvertStubData(const py::object &obj, ValuePtr *data, bool use_signature, const TypePtr &dtype,
                     bool forbid_reuse) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "The value pointer should not be null.";
    return false;
  }
  ValuePtr converted = nullptr;
  const auto &convert_funcs = GetStubDataConvertFuncs();
  for (auto &convert_func : convert_funcs) {
    if (convert_func->Matched(obj)) {
      converted = convert_func->ConvertPyObject(obj, use_signature, dtype);
      *data = converted;
      return converted != nullptr;
    }
  }
  return ConvertData(obj, data, use_signature, dtype, forbid_reuse);
}

FuncGraphPtr MakeReusingGraph(const FuncGraphPtr &base_graph) {
  static int order = 0;
  base_graph->set_attr(FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER, MakeValue(++order));
  base_graph->debug_info()->set_name("CR_" + base_graph->debug_info()->name());
  MS_LOG(INFO) << "Lazy inline reusing graph: " << base_graph->ToString()
               << ", args: " << base_graph->parameters().size() << ", parse order: " << order;
  return base_graph;
}

FuncGraphPtr MakeCellFuncGraph(const py::object &obj, const std::string &obj_id, const FuncGraphPtr &reusing_graph) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  // Normalize the name.
  auto function_name = obj_id;
  std::replace(function_name.begin(), function_name.end(), '.', '_');
  std::replace(function_name.begin(), function_name.end(), '<', '_');
  std::replace(function_name.begin(), function_name.end(), '>', '_');
  func_graph->debug_info()->set_name(function_name);
  PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
  func_graph->set_python_obj(python_obj);
  func_graph->set_flag(FUNC_GRAPH_FLAG_PROXY_GRAPH, true);
  std::vector<AnfNodePtr> new_node_inputs;
  new_node_inputs.push_back(NewValueNode(reusing_graph));
  for (const auto &origin_param : reusing_graph->parameters()) {
    auto param = func_graph->add_parameter();
    param->set_debug_info(origin_param->debug_info());
    new_node_inputs.push_back(param);
  }
  AnfNodePtr out = func_graph->NewCNodeInOrder(new_node_inputs);
  func_graph->set_output(out);
  MS_LOG(INFO) << "Lazy inline cell: " << func_graph->ToString() << ", args: " << func_graph->parameters().size();
  return func_graph;
}

FuncGraphPtr ProcessLazyInline(const py::object &obj, const ValuePtrList &args_value_list,
                               const std::string &python_mod_get_parse_method, const std::string &obj_id,
                               const std::string &obj_key) {
  ValuePtr key_value = nullptr;
  FuncGraphPtr reusing_graph = nullptr;
  bool is_key_cache = data_converter::GetObjectValue(obj_key, &key_value);
  if (is_key_cache && key_value != nullptr && key_value->isa<FuncGraph>()) {
    MS_LOG(DEBUG) << "Get the cache data, obj: " << obj_key;
    reusing_graph = key_value->cast<FuncGraphPtr>();
  } else {
    auto base_graph = ParsePythonCode(obj, python_mod_get_parse_method, args_value_list);
    if (base_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    if (Parser::GetTopFuncGraph() == base_graph) {
      return base_graph;
    }
    PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
    base_graph->set_python_obj(python_obj);
    MS_LOG(DEBUG) << "Parse reusing function: " << reusing_graph->ToString();
    reusing_graph = MakeReusingGraph(base_graph);
    data_converter::CacheObjectValue(obj_key, reusing_graph);
  }
  // Let the original cell graph call the reusable graph.
  auto func_graph = MakeCellFuncGraph(obj, obj_id, reusing_graph);
  MS_LOG(DEBUG) << func_graph->ToString() << " calls " << reusing_graph->ToString();
  return func_graph;
}

// Convert data to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list,
                                const std::string &python_mod_get_parse_method, bool forbid_reuse) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_id = results[0] + python_mod_get_parse_method;
  std::string obj_key = results[1];
  FuncGraphPtr func_graph = nullptr;
  ValuePtr value = nullptr;
  bool is_debug = MsContext::GetInstance()->get_param<int>(MS_CTX_DEBUG_LEVEL) == kLevelDebug;
  bool is_cache = data_converter::GetObjectValue(obj_id, &value);
  if (!is_debug && is_cache && value != nullptr && value->isa<FuncGraph>()) {
    func_graph = value->cast<FuncGraphPtr>();
    if (!func_graph->dropped()) {
      bool has_forbid_reuse_attr = py::hasattr(obj, PYTHON_FUNCTION_FORBID_REUSE);
      if (forbid_reuse || has_forbid_reuse_attr) {
        return BasicClone(func_graph);
      }
      return func_graph;
    }
  }
  if (obj_key.find("lazy_inline") != obj_key.npos) {
    func_graph = ProcessLazyInline(obj, args_value_list, python_mod_get_parse_method, results[0], obj_key);
    if (func_graph == nullptr) {
      return nullptr;
    }
  } else {
    func_graph = ParsePythonCode(obj, python_mod_get_parse_method, args_value_list);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
  }

  data_converter::CacheObjectValue(obj_id, func_graph);
  if (!obj_key.empty() && python_mod_get_parse_method == PYTHON_MOD_GET_PARSE_METHOD) {
    data_converter::SetObjGraphValue(obj_key, func_graph);
  }

  PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
  func_graph->set_python_obj(python_obj);

  if (forbid_reuse) {
    // The function may be set recomputed in parse.
    if (!data_converter::IsCellInstance(obj)) {
      RemoveRecomputeScope(func_graph);
    }
    // Return the clone graph because the graph may be set recomputed later.
    return BasicClone(func_graph);
  }

  return func_graph;
}

ValuePtr GetArgDefaultValue(const std::string &prim_name, const std::string &arg_name) {
  py::module mod = py::module::import(PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE);
  if (!py::hasattr(mod, PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not found " << PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT << "in "
                               << PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE << ".";
  }
  py::dict op_default_dict = mod.attr(PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT);
  if (!op_default_dict.contains(py::str(prim_name))) {
    return nullptr;
  }
  py::dict prim_default_dict = op_default_dict[py::str(prim_name)];
  if (!prim_default_dict.contains(py::str(arg_name))) {
    return nullptr;
  }
  auto default_value = prim_default_dict[py::str(arg_name)];
  ValuePtr converted_ret = nullptr;
  bool converted = ConvertData(default_value, &converted_ret);
  if (!converted) {
    const std::string &default_name = py::str(default_value);
    MS_EXCEPTION(ValueError) << "For Operator[" << prim_name << "], '" << default_name
                             << "' is not supported as the default value for '" << arg_name << "'.";
  }
  return converted_ret;
}

namespace data_converter {
static mindspore::HashMap<std::string, ValuePtr> object_map_;

static mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> object_graphs_map_;

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data) {
  object_graphs_map_[obj_key].push_back(data);
  MS_LOG(DEBUG) << "Set func graph size: " << object_graphs_map_.size();
}

const mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs() {
  MS_LOG(DEBUG) << "Obj graphs size: " << object_graphs_map_.size();
  return object_graphs_map_;
}

void CacheObjectValue(const std::string &obj_key, const ValuePtr &data) { object_map_[obj_key] = data; }

bool GetObjectValue(const std::string &obj_key, ValuePtr *const data) {
  if (object_map_.count(obj_key) != 0) {
    *data = object_map_[obj_key];
    return true;
  }
  return false;
}

std::vector<std::string> GetObjKey(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::tuple obj_tuple = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_KEY, obj);
  if (obj_tuple.size() != 2) {
    MS_LOG(INTERNAL_EXCEPTION) << "The function of \'get_obj_key()\' must return 2 elements";
  }
  return {py::cast<std::string>(obj_tuple[0]), py::cast<std::string>(obj_tuple[1])};
}

// Get obj detail type
ResolveType GetObjType(const py::object &obj) {
  try {
    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    auto obj_type = ResolveType(python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_TYPE, obj).cast<int32_t>());
    return obj_type;
  } catch (const py::error_already_set &ex) {
    MS_LOG(ERROR) << "Meet a exception from Python when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  } catch (const py::type_error &ex) {
    MS_LOG(ERROR) << "Meet a exception when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  }
}

// Get class instance detail type.
ClassInstanceType GetClassInstanceType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto class_type =
    ClassInstanceType(python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_CLASS_INSTANCE_TYPE, obj).cast<int32_t>());
  return class_type;
}

// Check if the object is Cell instance.
bool IsCellInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  return class_type == CLASS_INSTANCE_TYPE_CELL;
}

// Check if the object is Numpy Array instance.
bool IsNumpyArrayInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  return class_type == CLASS_INSTANCE_TYPE_NUMPY_ARRAY;
}

// Check if the object is MsClass instance.
bool IsMsClassInstance(const py::object &obj) { return py::hasattr(obj, PYTHON_MS_CLASS); }

// Check if the object is jit forbidden api.
bool IsJITForbiddenAPI(const py::object &obj) { return py::hasattr(obj, PYTHON_JIT_FORBIDDEN); }

// Check if the object is class type.
bool IsClassType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_CLASS_TYPE, obj).cast<bool>();
}

// Create the python class instance.
py::object CreatePythonObject(const py::object &type, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` maybe a tuple(*args), tuple(**kwargs), or tuple(*args, **kwargs).
  return args_kwargs.empty() ? python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type)
                             : python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type, args_kwargs);
}

// Call the python script string.
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` is a tuple(dict(global), dict(local)).
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_EVAL_PY_SCRIPT, script, args_kwargs);
}

// Get the ids of python script string.
py::set GetPythonScriptIdAttrs(const py::object &script) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_SCRIPT_ID_ATTRS, script);
}

ValuePtr PyDataToValue(const py::object &obj) {
  py::object to_convert = obj;
  ValuePtr value = nullptr;
  (void)ConvertData(to_convert, &value);
  return value;
}

ValuePtr PyDataToStubNode(const py::object &obj) {
  py::object to_convert = obj;
  ValuePtr value = nullptr;
  (void)ConvertStubData(to_convert, &value);
  return value;
}

void ClearObjectCache() {
  object_map_.clear();
  object_graphs_map_.clear();
}
}  // namespace data_converter

ValuePtr DataConverter::ConvertData(const py::object &obj) {
  const auto &convert_funcs = GetDataConvertFuncs();
  for (auto &convert_func : convert_funcs) {
    if (convert_func->Matched(obj)) {
      return convert_func->ConvertPyObject(obj, use_signature_, dtype_, args_value_list_);
    }
  }
  return ConvertOtherObj(obj, forbid_reuse_);
}

inline ValuePtr ConvertPythonFloatToScalarValue(double value) {
  auto ret = std::make_shared<FP32Imm>(static_cast<float>(value));
  ret->set_prim_value(value);
  return ret;
}

ValuePtr ConvertBool(const py::object &obj) {
  if (!py::isinstance<py::bool_>(obj)) {
    return nullptr;
  }
  return PyCast<BoolImm, bool>(obj);
}

ValuePtr ConvertInt(const py::object &obj) {
  // bool is also an instance of py::int_
  if (py::isinstance<py::bool_>(obj) || !py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return ConvertIntegerWithType(obj);
}

ValuePtr ConvertFloat(const py::object &obj) {
  if (!py::isinstance<py::float_>(obj)) {
    return nullptr;
  }
  return ConvertFloatWithType(obj);
}

ValuePtr ConvertNumber(const py::object &obj) {
  if (py::isinstance<py::bool_>(obj)) {
    return PyCast<BoolImm, bool>(obj);
  }

  if (py::isinstance<py::int_>(obj)) {
    return ConvertIntegerWithType(obj);
  }

  if (py::isinstance<py::float_>(obj)) {
    return ConvertFloatWithType(obj);
  }

  return nullptr;
}

ValuePtr ConvertTensor(const py::object &obj) {
  if (IsStubTensor(obj)) {
    return PyStubNodeCast(obj);
  }

  if (!py::isinstance<mindspore::tensor::Tensor>(obj)) {
    return nullptr;
  }

  return ObjCast<TensorPtr>(obj);
}

TensorPtr ConvertTensorValue(const py::object &obj) {
  // The difference between the new ConvertTensorValue function and the existing ConvertTensor is:
  // If the obj a StubNode, it must be called the WaitValue to convert to a Tensor.
  if (IsStubTensor(obj)) {
    auto py_stub = py::getattr(obj, stub::PY_ATTR_STUB);
    auto stub = py_stub.cast<stub::StubNodePtr>();
    if (stub == nullptr) {
      return py::getattr(obj, stub::PY_ATTR_TENSOR).cast<tensor::TensorPtr>();
    }
    auto value = stub->WaitValue();
    auto tensor = value->cast<TensorPtr>();
    if (tensor == nullptr) {
      // BaseTensor should convert to Tensor for Graph mode
      auto base_tensor = value->cast<BaseTensorPtr>();
      auto real_tensor = std::make_shared<Tensor>(*base_tensor);
      stub->SetValue(real_tensor);
      return real_tensor;
    }
    return tensor;
  }
  if (!py::isinstance<mindspore::tensor::Tensor>(obj)) {
    return nullptr;
  }
  return obj.cast<TensorPtr>();
}

static inline void *GetTensorDataPtr(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &device_address = tensor->device_address();
  if (device_address != nullptr) {
    // Before get data, sync form device address should be performed first
    tensor->data_sync();
  }
  return tensor->data_c();
}

ValuePtr ConvertStr(const py::object &obj) {
  if (!py::isinstance<py::str>(obj)) {
    return nullptr;
  }
  return PyCast<StringImm, string>(obj);
}

ValuePtr ConvertAny(const py::object &obj) { return parse::data_converter::PyDataToStubNode(obj); }

ValuePtr ConvertDtype(const py::object &obj) {
  if (!py::isinstance<mindspore::Type>(obj)) {
    MS_LOG(EXCEPTION) << "Get arg is not mindspore type " << py::str(obj);
  }
  return obj.cast<TypePtr>();
}

template <typename TS, typename TD, OpDefConvertFunc func>
ValuePtr ConvertSequence(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }
  auto seq = obj.cast<TS>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < seq.size(); ++it) {
    auto out = func(seq[it]);
    if (out == nullptr) {
      return nullptr;
    }
    value_list.emplace_back(out);
  }
  return std::make_shared<TD>(value_list);
}

template <typename T, OpDefConvertFunc func>
ValuePtr ConvertSingleElementToSequence(const py::object &obj) {
  auto value = func(obj);
  if (value == nullptr) {
    return nullptr;
  }
  std::vector<ValuePtr> value_list{value};
  return std::make_shared<T>(std::move(value_list));
}

template <typename T1, typename T2>
ValuePtr ConvertSingleElementToTensor(const py::object &obj) {
  if (!py::isinstance<T1>(obj)) {
    return nullptr;
  }

  auto v = py::cast<T2>(obj);
  return std::make_shared<tensor::Tensor>(v);
}

ValuePtr ConvertNumberToTensor(const py::object &obj) {
  if (py::isinstance<py::bool_>(obj)) {
    auto v = py::cast<bool>(obj);
    return std::make_shared<tensor::Tensor>(v);
  }

  if (py::isinstance<py::int_>(obj)) {
    auto v = py::cast<int64_t>(obj);
    return std::make_shared<tensor::Tensor>(v);
  }

  if (py::isinstance<py::float_>(obj)) {
    auto v = py::cast<pyfloat>(obj);
    return std::make_shared<tensor::Tensor>(v);
  }

  return nullptr;
}

template <typename TS, typename TSE, typename TDE>
ValuePtr ConvertSequenceToTensor(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }

  auto seq = obj.cast<TS>();
  if (seq.size() == 0) {
    return nullptr;
  }

  std::vector<TDE> value_list;
  for (size_t it = 0; it < seq.size(); ++it) {
    if (!py::isinstance<TSE>(seq[it])) {
      return nullptr;
    }

    auto value = py::cast<TDE>(seq[it]);
    value_list.emplace_back(value);
  }

  return std::make_shared<tensor::Tensor>(value_list);
}

template <typename TS>
ValuePtr ConvertSequenceBoolToTensor(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }

  auto seq = obj.cast<TS>();
  if (seq.size() == 0) {
    return nullptr;
  }

  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, ShapeVector({static_cast<int64_t>(seq.size())}));
  auto data = static_cast<bool *>(tensor->data_c());
  for (size_t it = 0; it < seq.size(); ++it) {
    if (!py::isinstance<py::bool_>(seq[it])) {
      return nullptr;
    }

    auto value = py::cast<bool>(seq[it]);
    data[it] = value;
  }

  return tensor;
}

template <typename TD, typename TDE, typename IMMTYPE, TypeId tid>
ValuePtr ConvertTensorToSequence(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << obj.get_type() << "] to Tensor.";
    return nullptr;
  }

  auto data_type = tensor->data_type();
  // Since the dst object type is only, once the src object is validated as Tensor, the other converting errors should
  // be thrown. There is no other paths for this case to run successfully.
  if (data_type != tid) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Sequence with type "
                  << TypeIdToString(tid) << ".";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data = static_cast<TDE *>(GetTensorDataPtr(tensor));
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list;
  for (size_t i = 0; i < size; i++) {
    value_list.emplace_back(std::make_shared<IMMTYPE>(data[i]));
  }
  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceInt(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << obj.get_type() << "] to Tensor.";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data_type = tensor->data_type();
  if (data_type != kNumberTypeInt64 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Int Sequence.";
    return nullptr;
  }
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list;
  if (data_type == kNumberTypeInt64) {
    auto data = static_cast<int64_t *>(GetTensorDataPtr(tensor));
    std::transform(data, data + size, std::back_inserter(value_list),
                   [](int64_t num) { return std::make_shared<Int64Imm>(num); });
  } else {
    auto data = static_cast<int32_t *>(GetTensorDataPtr(tensor));
    std::transform(data, data + size, std::back_inserter(value_list),
                   [](int32_t num) { return std::make_shared<Int64Imm>(num); });
  }
  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceFloat(const py::object &obj) {
  auto float_tensor = ConvertTensorValue(obj);
  if (float_tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << obj.get_type() << "] to Tensor.";
    return nullptr;
  }

  auto data_type = float_tensor->data_type();
  if (data_type != kNumberTypeFloat64) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Float64 Sequence.";
    return nullptr;
  }

  auto shape = float_tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data = static_cast<double *>(GetTensorDataPtr(float_tensor));
  auto size = float_tensor->DataSize();
  std::vector<ValuePtr> value_list(size);
  for (size_t i = 0; i < size; i++) {
    value_list.emplace_back(ConvertPythonFloatToScalarValue(data[i]));
  }

  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceAny(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << obj.get_type() << "] to Tensor.";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data_type = tensor->data_type();
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list(size);
  if (data_type == kNumberTypeInt64) {
    auto data = static_cast<int64_t *>(GetTensorDataPtr(tensor));
    for (size_t i = 0; i < size; i++) {
      value_list.emplace_back(std::make_shared<Int64Imm>(data[i]));
    }
  } else if (data_type == kNumberTypeFloat64) {
    auto data = static_cast<double *>(GetTensorDataPtr(tensor));
    for (size_t i = 0; i < size; i++) {
      value_list.emplace_back(ConvertPythonFloatToScalarValue(data[i]));
    }
  } else if (data_type == kNumberTypeBool) {
    auto data = static_cast<bool *>(GetTensorDataPtr(tensor));
    for (size_t i = 0; i < size; i++) {
      value_list.emplace_back(std::make_shared<BoolImm>(data[i]));
    }
  } else {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << " to sequence.";
    return nullptr;
  }

  return std::make_shared<TD>(value_list);
}

ValuePtr ConvertTensorToInt(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "Can only convert tensor with one element to int, but got " << tensor->ToString();
    return nullptr;
  }
  if (tensor->data_type() == kNumberTypeInt64) {
    return std::make_shared<Int64Imm>(static_cast<int64_t *>(GetTensorDataPtr(tensor))[0]);
  } else if (tensor->data_type() == kNumberTypeInt32) {
    return std::make_shared<Int64Imm>(static_cast<int32_t *>(GetTensorDataPtr(tensor))[0]);
  } else {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to int";
    return nullptr;
  }
}

ValuePtr ConvertTensorToFloat(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "Can only convert tensor with one element to float, but got " << tensor->ToString();
    return nullptr;
  }
  if (tensor->data_type() != kNumberTypeFloat64) {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to float";
    return nullptr;
  }
  return ConvertPythonFloatToScalarValue(static_cast<double *>(GetTensorDataPtr(tensor))[0]);
}

ValuePtr ConvertTensorToBool(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->data_type() != kNumberTypeBool) {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to bool";
    return nullptr;
  }
  return std::make_shared<BoolImm>(static_cast<bool *>(GetTensorDataPtr(tensor))[0]);
}

ValuePtr ConvertTensorToNumber(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_EXCEPTION(ValueError) << "Can only convert tensor with one element to number, but got " << tensor->ToString();
  }

  switch (tensor->data_type()) {
    case kNumberTypeBool:
      return std::make_shared<BoolImm>(static_cast<bool *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeInt64:
      return std::make_shared<Int64Imm>(static_cast<int64_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeInt32:
      return std::make_shared<Int32Imm>(static_cast<int32_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeInt16:
      return std::make_shared<Int64Imm>(static_cast<int16_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeInt8:
      return std::make_shared<Int64Imm>(static_cast<int8_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeUInt64:
      return std::make_shared<Int64Imm>(static_cast<uint64_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeUInt32:
      return std::make_shared<Int64Imm>(static_cast<uint32_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeUInt16:
      return std::make_shared<Int64Imm>(static_cast<uint16_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeUInt8:
      return std::make_shared<Int64Imm>(static_cast<uint8_t *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeFloat64:
      return ConvertPythonFloatToScalarValue(static_cast<double *>(GetTensorDataPtr(tensor))[0]);
    case kNumberTypeFloat32:
      return ConvertPythonFloatToScalarValue(static_cast<float *>(GetTensorDataPtr(tensor))[0]);
    default:
      MS_EXCEPTION(TypeError) << "Can not convert " << tensor->ToString() << " to number";
  }
}

ValuePtr ConvertBoolOrIntToFloat(const py::object &obj) {
  // bool is also an instance of py::int_
  if (!py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return ConvertFloatWithType(obj);
}

static const std::unordered_map<int32_t, OpDefConvertFunc> kConverters = {
  // convert functions without type_cast
  {(int32_t)mindspore::ops::DT_BOOL, ConvertBool},
  {(int32_t)mindspore::ops::DT_INT, ConvertInt},
  {(int32_t)mindspore::ops::DT_FLOAT, ConvertFloat},
  {(int32_t)mindspore::ops::DT_NUMBER, ConvertNumber},
  {(int32_t)mindspore::ops::DT_TENSOR, ConvertTensor},
  {(int32_t)mindspore::ops::DT_STR, ConvertStr},
  {(int32_t)mindspore::ops::DT_ANY, ConvertAny},
  {(int32_t)mindspore::ops::DT_TYPE, ConvertDtype},
  {(int32_t)mindspore::ops::DT_TUPLE_BOOL, ConvertSequence<py::tuple, ValueTuple, ConvertBool>},
  {(int32_t)mindspore::ops::DT_TUPLE_INT, ConvertSequence<py::tuple, ValueTuple, ConvertInt>},
  {(int32_t)mindspore::ops::DT_TUPLE_FLOAT, ConvertSequence<py::tuple, ValueTuple, ConvertFloat>},
  {(int32_t)mindspore::ops::DT_TUPLE_NUMBER, ConvertSequence<py::tuple, ValueTuple, ConvertNumber>},
  {(int32_t)mindspore::ops::DT_TUPLE_TENSOR, ConvertSequence<py::tuple, ValueTuple, ConvertTensor>},
  {(int32_t)mindspore::ops::DT_TUPLE_STR, ConvertSequence<py::tuple, ValueTuple, ConvertStr>},
  {(int32_t)mindspore::ops::DT_TUPLE_ANY, ConvertSequence<py::tuple, ValueTuple, ConvertAny>},
  {(int32_t)mindspore::ops::DT_LIST_BOOL, ConvertSequence<py::list, ValueList, ConvertBool>},
  {(int32_t)mindspore::ops::DT_LIST_INT, ConvertSequence<py::list, ValueList, ConvertInt>},
  {(int32_t)mindspore::ops::DT_LIST_FLOAT, ConvertSequence<py::list, ValueList, ConvertFloat>},
  {(int32_t)mindspore::ops::DT_LIST_NUMBER, ConvertSequence<py::list, ValueList, ConvertNumber>},
  {(int32_t)mindspore::ops::DT_LIST_TENSOR, ConvertSequence<py::list, ValueList, ConvertTensor>},
  {(int32_t)mindspore::ops::DT_LIST_STR, ConvertSequence<py::list, ValueList, ConvertStr>},
  {(int32_t)mindspore::ops::DT_LIST_ANY, ConvertSequence<py::list, ValueList, ConvertAny>},

  // TypeCast1: convert single element to sequence
  {CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_TUPLE_INT),
   ConvertSingleElementToSequence<ValueTuple, ConvertNumber>},
  {CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_LIST_INT),
   ConvertSingleElementToSequence<ValueList, ConvertNumber>},
  {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_TUPLE_INT),
   ConvertSingleElementToSequence<ValueTuple, ConvertInt>},
  {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_LIST_INT),
   ConvertSingleElementToSequence<ValueList, ConvertInt>},
  {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_TUPLE_INT),
   ConvertSingleElementToSequence<ValueTuple, ConvertFloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_LIST_INT),
   ConvertSingleElementToSequence<ValueList, ConvertFloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_TUPLE_INT),
   ConvertSingleElementToSequence<ValueTuple, ConvertBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_LIST_INT),
   ConvertSingleElementToSequence<ValueList, ConvertBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_ANY, mindspore::ops::DT_TUPLE_ANY),
   ConvertSingleElementToSequence<ValueTuple, ConvertAny>},
  {CombineTypesForTypeCast(mindspore::ops::DT_ANY, mindspore::ops::DT_LIST_ANY),
   ConvertSingleElementToSequence<ValueList, ConvertAny>},

  // TypeCast2: convert sequence to sequence, such as py::tuple to ValueList
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_INT, mindspore::ops::DT_LIST_INT),
   ConvertSequence<py::tuple, ValueList, ConvertInt>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_FLOAT, mindspore::ops::DT_LIST_FLOAT),
   ConvertSequence<py::tuple, ValueList, ConvertFloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_BOOL, mindspore::ops::DT_LIST_BOOL),
   ConvertSequence<py::tuple, ValueList, ConvertBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_ANY, mindspore::ops::DT_LIST_ANY),
   ConvertSequence<py::tuple, ValueList, ConvertAny>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_TENSOR, mindspore::ops::DT_LIST_TENSOR),
   ConvertSequence<py::tuple, ValueList, ConvertTensor>},

  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_INT, mindspore::ops::DT_TUPLE_INT),
   ConvertSequence<py::list, ValueTuple, ConvertInt>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_FLOAT, mindspore::ops::DT_TUPLE_FLOAT),
   ConvertSequence<py::list, ValueTuple, ConvertFloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_BOOL, mindspore::ops::DT_TUPLE_BOOL),
   ConvertSequence<py::list, ValueTuple, ConvertBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_ANY, mindspore::ops::DT_TUPLE_ANY),
   ConvertSequence<py::list, ValueTuple, ConvertAny>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_TENSOR, mindspore::ops::DT_TUPLE_TENSOR),
   ConvertSequence<py::list, ValueTuple, ConvertAny>},

  // TypeCast3: convert single element to Tensor
  {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_TENSOR),
   ConvertSingleElementToTensor<py::int_, pyint>},
  {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_TENSOR),
   ConvertSingleElementToTensor<py::float_, pyfloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_TENSOR),
   ConvertSingleElementToTensor<py::bool_, bool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_TENSOR), ConvertNumberToTensor},

  // TypeCast4: convert between sequence and tensor
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_INT, mindspore::ops::DT_TENSOR),
   ConvertSequenceToTensor<py::tuple, py::int_, pyint>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_FLOAT, mindspore::ops::DT_TENSOR),
   ConvertSequenceToTensor<py::tuple, py::float_, pyfloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_BOOL, mindspore::ops::DT_TENSOR),
   ConvertSequenceBoolToTensor<py::tuple>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_INT, mindspore::ops::DT_TENSOR),
   ConvertSequenceToTensor<py::list, py::int_, pyint>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_FLOAT, mindspore::ops::DT_TENSOR),
   ConvertSequenceToTensor<py::list, py::float_, pyfloat>},
  {CombineTypesForTypeCast(mindspore::ops::DT_LIST_BOOL, mindspore::ops::DT_TENSOR),
   ConvertSequenceBoolToTensor<py::list>},

  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_INT),
   ConvertTensorToSequenceInt<ValueTuple>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_FLOAT),
   ConvertTensorToSequenceFloat<ValueTuple>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_BOOL),
   ConvertTensorToSequence<ValueTuple, bool, BoolImm, kNumberTypeBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_BOOL),
   ConvertTensorToSequenceAny<ValueTuple>},

  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_INT),
   ConvertTensorToSequenceInt<ValueList>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_FLOAT),
   ConvertTensorToSequenceFloat<ValueList>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_BOOL),
   ConvertTensorToSequence<ValueList, bool, BoolImm, kNumberTypeBool>},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_BOOL),
   ConvertTensorToSequenceAny<ValueList>},

  // TypeCast5: convert tensor to single element
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_INT), ConvertTensorToInt},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_FLOAT), ConvertTensorToFloat},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_BOOL), ConvertTensorToBool},
  {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_NUMBER), ConvertTensorToNumber},

  // TypeCas6: convert int/bool to float
  {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_FLOAT), ConvertBoolOrIntToFloat},
};

OpDefConvertFunc GetConverterByType(int32_t dtype) {
  auto it = kConverters.find(dtype);
  if (it == kConverters.end()) {
    if ((dtype >> kTypeShiftBits) == 0) {
      MS_LOG(EXCEPTION) << "Can not find converter for dtype[" << ops::EnumToString(static_cast<ops::OP_DTYPE>(dtype))
                        << "].";
    } else {
      MS_LOG(EXCEPTION) << "Can not find converter for src_type["
                        << ops::EnumToString(static_cast<ops::OP_DTYPE>(dtype >> kTypeShiftBits)) << "] and dst_type["
                        << ops::EnumToString(static_cast<ops::OP_DTYPE>(dtype & kDstMask)) << "].";
    }
  }

  return it->second;
}
}  // namespace parse
}  // namespace mindspore
