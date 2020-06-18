/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/parse/data_converter.h"
#include <unordered_map>
#include <map>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include <list>
#include "pipeline/parse/resolve.h"
#include "pipeline/parse/python_adapter.h"
#include "operator/ops.h"
#include "operator/composite/composite.h"
#include "ir/func_graph_cloner.h"
#include "utils/symbolic.h"
#include "utils/context/ms_context.h"
#include "debug/trace.h"
#include "optimizer/ad/grad.h"

namespace mindspore {
namespace parse {
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaTensorPtr = mindspore::tensor::MetaTensorPtr;

FuncGraphPtr ConvertToBpropCut(const py::object &obj) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_key = results[0];
  py::function bprop_func = py::getattr(obj, CUSTOM_BPROP_NAME);

  auto bprop_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto fake_bprop = std::make_shared<PrimitivePy>("bprop_cut", py::object());
  fake_bprop->set_hook(bprop_func);
  (void)fake_bprop->AddAttr(CUSTOM_BPROP_NAME, MakeValue(true));
  outputs.push_back(NewValueNode(fake_bprop));

  py::object code_obj = py::getattr(bprop_func, "__code__");
  size_t inputs_num = py::cast<int>(py::getattr(code_obj, "co_argcount")) - 3;
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = bprop_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = bprop_graph->add_parameter();
  auto p2 = bprop_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  bprop_graph->set_output(bprop_graph->NewCNode(outputs));
  data_converter::SetObjGraphValue(obj_key, bprop_graph);
  return bprop_graph;
}

namespace {
bool ConvertTuple(const py::object &obj, ValuePtr *const data, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  py::tuple tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(tuple[it], &out, use_signature);
    if (!success) {
      return false;
    }
    value_list.push_back(out);
  }
  *data = std::make_shared<ValueTuple>(value_list);

  return true;
}

bool ConvertList(const py::object &obj, ValuePtr *const data, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";

  py::list list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return false;
    }
    value_list.push_back(out);
  }
  *data = std::make_shared<ValueList>(value_list);
  return true;
}

bool ConvertCellList(const py::object &obj, ValuePtr *const data, bool use_signature) {
  MS_LOG(DEBUG) << "Converting cell list";
  py::sequence list = obj;
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return false;
    }
    value_list.push_back(out);
  }
  *data = std::make_shared<ValueTuple>(value_list);
  return true;
}

bool ConvertDict(const py::object &obj, ValuePtr *data, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python dict";

  py::dict dict_values = obj.cast<py::dict>();
  std::vector<std::pair<std::string, ValuePtr>> key_values;
  for (auto item : dict_values) {
    if (!py::isinstance<py::str>(item.first)) {
      MS_LOG(EXCEPTION) << "The key of dict is only support str.";
    }
    std::string key = py::str(item.first);
    ValuePtr out = nullptr;
    bool success = ConvertData(dict_values[item.first], &out, use_signature);
    if (!success) {
      return false;
    }
    key_values.emplace_back(key, out);
  }
  *data = std::make_shared<ValueDictionary>(key_values);
  return true;
}

void ConvertNameSpace(const py::object &obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting python module";
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object module_namespace = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MODULE_NAMESPACE, obj);
  *data = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_MODULE, py::cast<py::module>(module_namespace));
}

void ConvertDataClass(py::object obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting dataclass";
  // Maybe the obj is dataclass define
  auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
  // desc has format "<class xxxx>", strip the '<' and '>' by offset 1;
  *data = std::make_shared<ClassObject>(obj, std::string(desc.begin() + 1, desc.end() - 1));
}

bool ConvertPrimitive(py::object obj, ValuePtr *const data, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting primitive object";

  // need check the primitive is class type or instance
  auto obj_type = data_converter::GetObjType(obj);
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1;
    *data = std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  } else {
    auto primitive = obj.cast<PrimitivePyPtr>();
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Resolve Primitive error, get ptr is null";
      return false;
    }
    if (py::hasattr(obj, "__setattr_flag__")) {
      if (py::hasattr(obj, "_clone")) {
        auto clone_fn = obj.attr("_clone");
        py::object new_obj = clone_fn();
        primitive = new_obj.cast<PrimitivePyPtr>();
      }
    }
    if (use_signature) {
      *data = std::make_shared<prim::DoSignaturePrimitive>(primitive->name(), primitive);
    } else {
      *data = primitive;
    }
  }
  return true;
}

bool ConvertMetaFuncGraph(const py::object &obj, ValuePtr *const data, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting MetaFuncGraph object";
  auto meta = obj.cast<MetaFuncGraphPtr>();
  if (meta == nullptr) {
    MS_LOG(ERROR) << "Resolve MetaFuncGraph error, get ptr is null";
    return false;
  }
  if (use_signature) {
    *data = std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta);
  } else {
    *data = meta;
  }
  return true;
}

bool ConvertDataType(const py::object &obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting type object";
  auto typeptr = obj.cast<TypePtr>();
  if (typeptr == nullptr) {
    MS_LOG(ERROR) << "Resolve TypePtr error, get ptr is null";
    return false;
  }
  *data = typeptr;
  return true;
}

bool ConvertMetaTensor(const py::object &obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting MetaTensor object.";

  auto m_tensor = obj.cast<MetaTensorPtr>();
  if (m_tensor == nullptr) {
    MS_LOG(ERROR) << "Resolve MetaTensor error, get ptr is null.";
    return false;
  }
  *data = m_tensor;
  return true;
}

bool ConvertTensor(const py::object &obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting tensor object";

  auto m_tensor = obj.cast<TensorPtr>();
  if (m_tensor == nullptr) {
    MS_LOG(ERROR) << "Resolve Tensor error, get ptr is null";
    return false;
  }
  *data = m_tensor;
  return true;
}

bool ConvertSlice(const py::object &obj, ValuePtr *const data) {
  MS_LOG(DEBUG) << "Converting slice object";

  py::slice slice_obj = obj.cast<py::slice>();
  auto convert_func = [obj](std::string attr) -> ValuePtr {
    auto py_attr = py::getattr(obj, attr.c_str());
    if (py::isinstance<py::none>(py_attr)) {
      return kNone;
    } else if (py::isinstance<py::int_>(py_attr)) {
      int value = py::cast<int>(py_attr);
      return MakeValue(value);
    } else {
      MS_LOG(EXCEPTION) << "Slice should contain only int or none";
    }
  };
  ValuePtr start = convert_func("start");
  ValuePtr stop = convert_func("stop");
  ValuePtr step = convert_func("step");
  *data = std::make_shared<ValueSlice>(start, stop, step);
  return true;
}

bool ConvertCellObjToFuncGraph(py::object obj, ValuePtr *const data) {
  FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return false;
  }
  // if the cell object has specified bprop, it has user-defined bprop function parse and record it
  if (py::hasattr(obj, CUSTOM_BPROP_NAME)) {
    FuncGraphPtr bprop_graph = nullptr;
    bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
    if (enable_bprop_debug) {
      bprop_graph = ConvertToBpropCut(obj);
    } else {
      bprop_graph = ConvertToFuncGraph(obj, PYTHON_MOD_GET_BPROP_METHOD);
    }
    if (bprop_graph != nullptr) {
      (void)func_graph->transforms().insert(std::make_pair(CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph)));
      (void)bprop_graph->transforms().insert(std::make_pair("primal", FuncGraphTransform(func_graph)));
      func_graph->set_flags(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    }
  }
  *data = func_graph;
  return true;
}

bool ConvertOtherObj(py::object obj, ValuePtr *const data) {
  auto obj_type = data_converter::GetObjType(obj);
  MS_LOG(DEBUG) << "Converting the object(" << ((std::string)py::str(obj)) << ") detail type: " << obj_type << " ";
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    MS_LOG(DEBUG) << "Resolve the class type, need create class instance.";
    std::string desc = py::str(obj);
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1;
    *data = std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
    return true;
  }
  if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD) {
    MS_LOG(DEBUG) << "Convert the obj to func graph, type is " << obj_type;
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return false;
    }
    *data = func_graph;
    return true;
  }
  if (obj_type == RESOLVE_TYPE_CLASS_INSTANCE) {
    // Create the namespace for common class instance
    // When the obj is Cell, default parse the 'construct'
    if (data_converter::IsCellInstance(obj)) {
      return ConvertCellObjToFuncGraph(obj, data);
    }

    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    py::object namespace_var = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, obj);
    *data = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
    return true;
  }
  MS_LOG(ERROR) << "Resolve type is invalid " << ((std::string)py::str(obj));
  return false;
}
}  // namespace

bool ConvertData(const py::object &obj, ValuePtr *const data, bool use_signature) {
  // check parameter valid
  if (data == nullptr) {
    MS_LOG(ERROR) << "Data is null pointer";
    return false;
  }

  bool ret = true;
  ValuePtr converted = nullptr;
  if (py::isinstance<py::none>(obj)) {
    converted = kNone;
  } else if (py::isinstance<py::bool_>(obj)) {
    converted = std::make_shared<BoolImm>(py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    converted = std::make_shared<Int32Imm>(py::cast<int>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    converted = std::make_shared<FP32Imm>(py::cast<float>(obj));
  } else if (py::isinstance<py::str>(obj)) {
    converted = std::make_shared<StringImm>(py::cast<std::string>(obj));
  } else if (py::isinstance<py::dict>(obj)) {
    ret = ConvertDict(obj, &converted, use_signature);
  } else if (py::isinstance<py::slice>(obj)) {
    ret = ConvertSlice(obj, &converted);
  } else if (py::isinstance<py::ellipsis>(obj)) {
    converted = kEllipsis;
  } else if (py::isinstance<py::tuple>(obj)) {
    ret = ConvertTuple(obj, &converted, use_signature);
  } else if (py::hasattr(obj, PYTHON_CELL_AS_LIST)) {
    ret = ConvertCellList(obj, &converted, use_signature);
  } else if (py::isinstance<py::list>(obj)) {
    ret = ConvertList(obj, &converted, use_signature);
  } else if (py::isinstance<py::module>(obj)) {
    ConvertNameSpace(obj, &converted);
  } else if (py::hasattr(obj, PYTHON_DATACLASS_FIELDS)) {
    ConvertDataClass(obj, &converted);
  } else if (py::hasattr(obj, PYTHON_PRIMITIVE_FLAG)) {
    ret = ConvertPrimitive(obj, &converted, use_signature);
  } else if (py::hasattr(obj, PYTHON_METAFUNCGRAPH_FLAG)) {
    ret = ConvertMetaFuncGraph(obj, &converted, use_signature);
  } else if (py::hasattr(obj, PYTHON_DTYPE_FLAG)) {
    ret = ConvertDataType(obj, &converted);
  } else if (py::hasattr(obj, PYTHON_TENSOR_FLAG)) {
    ret = ConvertTensor(obj, &converted);
  } else if (py::hasattr(obj, PYTHON_META_TENSOR_FLAG)) {
    ret = ConvertMetaTensor(obj, &converted);
  } else if (py::hasattr(obj, PYTHON_ENVINSTANCE_FLAG)) {
    std::shared_ptr<EnvInstance> env = obj.cast<std::shared_ptr<EnvInstance>>();
    converted = env;
  } else if (py::hasattr(obj, "__parameter__")) {
    auto to_convert = py::cast<py::object>(python_adapter::GetPyObjAttr(obj, "default_input"));
    ret = ConvertData(to_convert, &converted);
  } else {
    ret = ConvertOtherObj(obj, &converted);
  }

  *data = converted;
  return ret;
}

// convert data to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const std::string &python_mod_get_parse_method) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_id = results[0] + python_mod_get_parse_method;
  std::string obj_key = results[1];
  FuncGraphPtr func_graph = nullptr;
  Any value = Any();
  bool is_cache = data_converter::GetObjectValue(obj_id, &value);
  if (is_cache) {
    if (value.is<FuncGraphPtr>()) {
      MS_LOG(DEBUG) << "Get the cache data, obj = " << obj_id;
      func_graph = value.cast<FuncGraphPtr>();
      return func_graph;
    }
  }

  func_graph = ParsePythonCode(obj, python_mod_get_parse_method);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }

  data_converter::MakeProperNameToFuncGraph(func_graph, obj_id);
  data_converter::CacheObjectValue(obj_id, func_graph);
  if (obj_key != "") {
    MS_LOG(DEBUG) << "Add graph:" << obj_key << ", func_graph:" << func_graph->ToString();
    data_converter::SetObjGraphValue(obj_key, func_graph);
  }

  return func_graph;
}
namespace data_converter {
static std::unordered_map<std::string, Any> object_map_ = std::unordered_map<std::string, Any>();

static std::unordered_map<std::string, std::vector<FuncGraphPtr>> object_graphs_map_ =
  std::unordered_map<std::string, std::vector<FuncGraphPtr>>();

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data) {
  object_graphs_map_[obj_key].push_back(data);
  MS_LOG(DEBUG) << "Set func graph size:" << object_graphs_map_.size();
}

const std::unordered_map<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs() {
  MS_LOG(DEBUG) << "Obj size:" << object_graphs_map_.size();
  return object_graphs_map_;
}

void CacheObjectValue(const std::string &obj_key, const Any &data) { object_map_[obj_key] = data; }
bool GetObjectValue(const std::string &obj_key, Any *const data) {
  if (object_map_.count(obj_key)) {
    *data = object_map_[obj_key];
    return true;
  }
  return false;
}
std::vector<std::string> GetObjKey(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::tuple obj_tuple = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_KEY, obj);
  if (obj_tuple.size() != 2) {
    MS_LOG(EXCEPTION) << "Get_obj_key must return 2 elements";
  }
  return {py::cast<std::string>(obj_tuple[0]), py::cast<std::string>(obj_tuple[1])};
}

// get obj detail type
ResolveTypeDef GetObjType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto obj_type =
    ResolveTypeDef(python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_TYPE, obj).cast<int32_t>());
  return obj_type;
}

// get class instance detail type
ClassInstanceTypeDef GetClassInstanceType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto class_type =
    ClassInstanceTypeDef(python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_CLASS_INSTANCE_TYPE, obj).cast<int32_t>());
  return class_type;
}

// check the object is Cell Instance
bool IsCellInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  bool isCell = (class_type == CLASS_INSTANCE_TYPE_CELL);
  return isCell;
}

// create the python class instance
py::object CreatePythonObject(const py::object &type, const py::tuple &params) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object obj;
  if (params.size() == 0) {
    obj = python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_OBJ_INSTANCE, type);
  } else {
    obj = python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_OBJ_INSTANCE, type, params);
  }
  return obj;
}

// Generate an appropriate name and set to graph debuginfo
// character <> can not used in the dot file, so change to another symbol
void MakeProperNameToFuncGraph(const FuncGraphPtr &func_graph, std::string name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->debug_info());
  // set detail name info of function
  std::ostringstream oss;
  for (size_t i = 0; i < name.size(); i++) {
    if (name[i] == '<') {
      oss << "「";
    } else if (name[i] == '>') {
      oss << "」";
    } else {
      oss << name[i];
    }
  }
  func_graph->debug_info()->set_full_name(oss.str());
}

ValuePtr PyDataToValue(const py::object &obj) {
  py::object to_convert = obj;
  if (py::hasattr(obj, "__parameter__")) {
    to_convert = py::cast<py::object>(python_adapter::GetPyObjAttr(obj, "default_input"));
  }
  ValuePtr value = nullptr;
  (void)ConvertData(to_convert, &value);
  return value;
}

void ClearObjectCache() {
  object_map_.clear();
  object_graphs_map_.clear();
}
}  // namespace data_converter

static std::unordered_map<std::string, ClassPtr> g_dataClassToClass = {};

// parse dataclass to mindspore Class type
ClassPtr ParseDataClass(const py::object &cls_obj) {
  std::string cls_name = py::cast<std::string>(python_adapter::GetPyObjAttr(cls_obj, "__name__"));
  std::string cls_module = py::cast<std::string>(python_adapter::GetPyObjAttr(cls_obj, "__module__"));
  std::string cls = cls_module + "." + cls_name;
  auto iterator = g_dataClassToClass.find(cls);
  if (iterator != g_dataClassToClass.end()) {
    return iterator->second;
  }

  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  ClassAttrVector attributes;
  py::dict names = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_DATACLASS_ATTRS, cls_obj);
  for (auto &item : names) {
    TypePtr type_value = item.second.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(type_value);
    MS_LOG(DEBUG) << "(Name: " << py::cast<std::string>(item.first) << ", type: " << type_value->ToString() << ")";
    attributes.push_back(std::make_pair(py::cast<std::string>(item.first), type_value));
  }

  std::unordered_map<std::string, ValuePtr> methods_map;
  py::dict methods = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_DATACLASS_METHODS, cls_obj);
  for (auto &item : methods) {
    std::string fun_name = item.first.cast<std::string>();
    py::object obj = py::cast<py::object>(item.second);
    std::shared_ptr<PyObjectWrapper> method_obj = std::make_shared<PyObjectWrapper>(obj, fun_name);
    methods_map[fun_name] = method_obj;
  }

  std::shared_ptr<Class> me_class = std::make_shared<Class>(Named(cls_name), attributes, methods_map);
  // static Variable for cache
  // cppcheck-suppress unreadVariable
  g_dataClassToClass[cls] = me_class;

  return me_class;
}

void CleanDataClassToClassMap() { g_dataClassToClass.clear(); }
}  // namespace parse
}  // namespace mindspore
