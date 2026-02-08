/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/ps/parse/data_converter.h"

#include <cstdint>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include "include/utils/tensor_py.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/pipeline.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "ir/func_graph_cloner.h"
#include "ir/cell.h"
#include "ir/dtype.h"
#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "include/utils/pipeline/fallback.h"
#include "include/utils/utils.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/pynative/py_parse.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "ir/func_graph_flag.h"

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
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
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
  fake_bprop->SetHookFn(bprop_func, HookType::kCellCustomBprop);
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

    if (tensor::IsTensorPy(py_attr)) {
      return tensor::ConvertToTensor(py_attr);
    }
    MS_LOG(EXCEPTION) << "Attribute '" << attr << "' of " << py::str(obj)
                      << " should be int or Tensor with Int type but got " << py::str(py_attr);
  };
  ValuePtr start = convert_func(kSliceStart);
  ValuePtr stop = convert_func(kSliceStop);
  ValuePtr step = convert_func(kSliceStep);
  return std::make_shared<ValueSlice>(start, stop, step);
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

FuncGraphPtr CreateShardFuncGraph(const py::object &obj, const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Create Shard Node for Cell: " << py::str(obj) << ", func_graph: " << func_graph->ToString();

  FuncGraphPtr shard_graph = std::make_shared<FuncGraph>();
  for (size_t i = 0; i < func_graph->parameters().size(); i++) {
    shard_graph->add_parameter();
  }

  auto in_strategy = parse::data_converter::PyDataToValue(py::getattr(obj, CELL_IN_STRATEGY));
  auto out_strategy = parse::data_converter::PyDataToValue(py::getattr(obj, CELL_OUT_STRATEGY));
  MS_LOG(INFO) << "in_strategy: " << in_strategy->ToString() << ", out_strategy: " << out_strategy->ToString();

  std::vector<AnfNodePtr> shard_inputs{NewValueNode(prim::kPrimShard),
                                       NewValueNode(func_graph),
                                       NewValueNode(in_strategy),
                                       NewValueNode(out_strategy),
                                       NewValueNode(MakeValue(kAscendDevice)),
                                       NewValueNode(MakeValue<int64_t>(0))};
  auto shard_node = shard_graph->NewCNodeInOrder(shard_inputs);

  std::vector<AnfNodePtr> shard_node_inputs{shard_node};
  auto shard_graph_params = shard_graph->parameters();
  (void)std::copy(shard_graph_params.begin(), shard_graph_params.end(), std::back_inserter(shard_node_inputs));
  auto shard_out = shard_graph->NewCNodeInOrder(shard_node_inputs);
  shard_graph->set_output(shard_out);

  shard_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  MS_EXCEPTION_IF_NULL(shard_graph->debug_info());
  shard_graph->debug_info()->set_name(func_graph->debug_info()->name() + "_with_shard");

  return shard_graph;
}

void SetAttrForCell(const py::object &obj, const std::string &attr_name, FuncGraphPtr func_graph) {
  auto cell = py::cast<CellPtr>(obj);
  if (cell != nullptr && cell->HasAttr(attr_name)) {
    const auto &value = cell->GetAttr(attr_name);
    MS_EXCEPTION_IF_NULL(value);
    func_graph->set_attr(attr_name, value);
    MS_LOG(DEBUG) << "AddAttr " << attr_name << " to Cell, the value is " << value;
  }
}

ValuePtr ConvertCellObjToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list) {
  if (py::hasattr(obj, "construct")) {
    const auto &construct_obj = py::getattr(obj, "construct");
    bool graph_mode = GraphPipelineCompiling();
    if (py::hasattr(construct_obj, "__trace_func__") && !graph_mode) {
      return prim::kPrimTraceGraph;
    }
  }
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

  SetAttrForCell(obj, kAttrRandomOpSnapShot, func_graph);

  if (py::hasattr(obj, CELL_COMPILE_PHASE) && !py::getattr(obj, CELL_COMPILE_PHASE).is_none()) {
    SetAttrForCell(obj, CELL_COMPILE_PHASE, func_graph);
  }

  if (py::hasattr(obj, CELL_IN_STRATEGY)) {
    auto cell_in_strategy_obj = py::getattr(obj, CELL_IN_STRATEGY);
    if (!cell_in_strategy_obj.is_none()) {
      func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      FuncGraphPtr shard_graph = CreateShardFuncGraph(obj, func_graph);
      return shard_graph;
    }
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

void CheckAPI(const py::object &obj) {
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
        << "such as initialized in the method '__init__' before assigning.\n";
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
    // Check if the API is decoratored by @jit_view_unsupported.
    bool is_jit_view_unsupported_register = data_converter::IsJitViewUnSupportedAPI(obj);
    static std::map<std::string, bool> mint_warning_map;
    if (is_jit_view_unsupported_register && !mint_warning_map[obj_name]) {
      MS_LOG(WARNING) << "The mint interface " << obj_name
                      << " was called, and the operators under this interface have different view capabilities on "
                         "pynative and graph mode. Use this interface with caution in graph mode, as it may produce "
                         "unexpected results. For more information, please refer to: "
                         "https://www.mindspore.cn/docs/en/master/features/view.html\n";
      mint_warning_map[obj_name] = true;
    }
  }
}

ValuePtr ConvertEvent(const py::object &obj) {
  py::object event_obj_id = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  auto id = event_obj_id.cast<std::string>();
  MS_LOG(DEBUG) << "The id of event obj is: " << id;
  static std::mutex event_mutex;
  static std::map<std::string, uint32_t> event_map;
  static uint32_t event_id = 0;
  std::lock_guard<std::mutex> lock(event_mutex);
  auto it = event_map.find(id);
  if (it != event_map.end()) {
    return std::make_shared<Event>(static_cast<int32_t>(it->second));
  }
  if (event_id == UINT32_MAX) {
    MS_LOG(EXCEPTION) << "Event ID overflow!";
  }
  event_map[id] = event_id++;
  return std::make_shared<Event>(static_cast<int32_t>(event_map[id]));
}

ValuePtr ConvertOtherObj(const py::object &obj, bool forbid_reuse = false) {
  auto obj_type = data_converter::GetObjType(obj);
  MS_LOG(DEBUG) << "Converting the object(" << ((std::string)py::str(obj)) << ") detail type: " << obj_type << " ";
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    // Check JIT forbidden API and the view mint interface which is not supported in graph mode.
    CheckAPI(obj);
    MS_LOG(DEBUG) << "Resolve the class type, need create class instance.";
    std::string desc = py::str(obj);
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD || obj_type == RESOLVE_TYPE_BUILTIN_METHOD ||
      (obj_type == RESOLVE_TYPE_CLASS_INSTANCE && py::hasattr(obj, PYTHON_PARSE_METHOD))) {
    if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD ||
        obj_type == RESOLVE_TYPE_BUILTIN_METHOD) {
      // Check JIT forbidden API and the view mint interface which is not supported in graph mode.
      CheckAPI(obj);
      // Check if the function is from a third-party library.
      py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
      bool is_third_party_function =
        python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_FROM_THIRD_PARTY_LIBRARY, obj).cast<bool>();
      if (is_third_party_function) {
        MS_LOG(DEBUG) << "Converting the function from third-party library: " << py::str(obj);
        return std::make_shared<InterpretedObject>(obj);
      }
    }
    bool graph_mode = GraphPipelineCompiling();
    if (py::hasattr(obj, "__trace_func__") && !graph_mode) {
      return prim::kPrimTraceGraph;
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

  if (obj_type == RESOLVE_TYPE_EVENT) {
    MS_LOG(DEBUG) << "The obj is event.";
    return ConvertEvent(obj);
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
    std::make_shared<ByFuncDataConvertFunc>(IsNamedTuple, ConvertNamedTuple),
    std::make_shared<ByFuncDataConvertFunc>(tensor::IsTensorPy, ConvertTensorAndSyncCompiling),
    std::make_shared<ByAttrDataConvertFunc>(ConvertMsClass, PYTHON_MS_CLASS),
    std::make_shared<ByTypeDataConvertFunc<stub::TensorNode>>(ConvertTensorNode),
    std::make_shared<ByTypeDataConvertFunc<py::tuple>>(ConvertTuple),
    std::make_shared<ByTypeDataConvertFunc<py::list>>(ConvertList),
    std::make_shared<ByTypeDataConvertFunc<py::bool_>>(PyCast<BoolImm, bool>),
    std::make_shared<ByTypeDataConvertFunc<py::int_>>(ConvertIntegerWithType),
    std::make_shared<ByTypeDataConvertFunc<py::float_>>(ConvertFloatWithType),
    std::make_shared<ByTypeDataConvertFunc<py::str>>(PyCast<StringImm, string>),
    std::make_shared<ByTypeDataConvertFunc<py::none>>(kNone),
    std::make_shared<ByTypeDataConvertFunc<CSRTensor>>(ObjCast<CSRTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<COOTensor>>(ObjCast<COOTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<MapTensor>>(ObjCast<MapTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<py::ellipsis>>(kEllipsis),
    std::make_shared<ByTypeDataConvertFunc<py::module>>(ConvertModuleNameSpace),
    std::make_shared<ByTypeDataConvertFunc<Type>>(ObjCast<TypePtr>),
    std::make_shared<ByTypeDataConvertFunc<UMonad>>(ObjCast<UMonadPtr>),
    std::make_shared<ByTypeDataConvertFunc<IOMonad>>(ObjCast<IOMonadPtr>),
    std::make_shared<ByTypeDataConvertFunc<Functional>>(ObjCast<FunctionalPtr>),
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
    std::make_shared<ByTypeDataConvertFunc<stub::TensorNode>>(ObjCast<std::shared_ptr<stub::TensorNode>>),
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
    FuncGraphPtr base_graph = nullptr;
    {
      MS_LOG_TRY_CATCH_SCOPE;
      base_graph = ParsePythonCode(obj, python_mod_get_parse_method, args_value_list);
    }
    if (base_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    if (Parser::GetTopFuncGraph() == base_graph) {
      return base_graph;
    }
    PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
    base_graph->set_python_obj(python_obj);
    reusing_graph = MakeReusingGraph(base_graph);
    MS_EXCEPTION_IF_NULL(reusing_graph);
    MS_LOG(DEBUG) << "Parse reusing function: " << reusing_graph->ToString();
    data_converter::CacheObjectValue(obj_key, reusing_graph);
  }
  // Let the original cell graph call the reusable graph.
  auto func_graph = MakeCellFuncGraph(obj, obj_id, reusing_graph);
  MS_LOG(DEBUG) << func_graph->ToString() << " calls " << reusing_graph->ToString();
  return func_graph;
}

void UpdateReuseFlag(const FuncGraphPtr &func_graph) {
  // If graph is reused, mark every node in this graph.
  if (func_graph == nullptr) {
    return;
  }
  auto nodes = func_graph->nodes();
  for (const auto &node : nodes) {
    if (!node->isa<CNode>() || node->debug_info() == nullptr || node->debug_info()->trace_info() == nullptr ||
        node->debug_info()->trace_info()->debug_info() == nullptr) {
      continue;
    }
    node->debug_info()->trace_info()->debug_info()->set_is_reusing();
  }
}

// Convert data to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list,
                                const std::string &python_mod_get_parse_method, bool forbid_reuse) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_id = results[0] + python_mod_get_parse_method;
  std::string obj_key = results[1];
  FuncGraphPtr func_graph = nullptr;
  ValuePtr value = nullptr;
  bool is_debug = MsContext::GetInstance()->get_param<int>(MS_CTX_DEBUG_LEVEL) == kLevelDebug ||
                  common::GetCompileConfig("DEBUG_LEVEL") == "1";
  bool is_cache = data_converter::GetObjectValue(obj_id, &value);
  if (!is_debug && is_cache && value != nullptr && value->isa<FuncGraph>()) {
    func_graph = value->cast<FuncGraphPtr>();
    UpdateReuseFlag(func_graph);
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
    {
      MS_LOG_TRY_CATCH_SCOPE;
      func_graph = ParsePythonCode(obj, python_mod_get_parse_method, args_value_list);
    }
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

py::object GetPrimDefaultDict(const std::string &prim_name) {
  py::module mod = py::module::import(PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE);
  if (!py::hasattr(mod, PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not found " << PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT << "in "
                               << PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE << ".";
  }
  py::dict op_default_dict = mod.attr(PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT);
  if (!op_default_dict.contains(py::str(prim_name))) {
    return py::none();
  }
  return op_default_dict[py::str(prim_name)];
}

ValuePtr GetArgDefaultValue(const std::string &prim_name, const std::string &arg_name) {
  auto prim_default_dict = GetPrimDefaultDict(prim_name);
  if (py::isinstance<py::none>(prim_default_dict)) {
    return nullptr;
  }
  auto py_dict = prim_default_dict.cast<py::dict>();
  auto default_value = py_dict[py::str(arg_name)];
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

// Check whether the object is a mint interface that exhibits inconsistent view behavior
// between PYNATIVE mode and graph mode.
bool IsJitViewUnSupportedAPI(const py::object &obj) { return py::hasattr(obj, PYTHON_JIT_VIEW_UNSUPPORTED); }

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

ValuePtr PyObjToValue(const py::object &obj, bool stub) {
  ValuePtr converted_ret;
  if (stub) {
    converted_ret = parse::data_converter::PyDataToStubNode(obj);
  } else {
    converted_ret = parse::data_converter::PyDataToValue(obj);
  }
  if (converted_ret == nullptr) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << ConvertPyObjToString(obj);
  }
  return converted_ret;
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

namespace {
ValuePtr ConvertAny(const py::object &obj) { return parse::data_converter::PyDataToStubNode(obj); }

std::unordered_map<int32_t, OpDefConvertFunc> GetAnyConverters() {
  static const std::unordered_map<int32_t, OpDefConvertFunc> kAnyConverters = {
    {static_cast<int32_t>(mindspore::ops::DT_ANY), ConvertAny},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_ANY), py_parse::ConvertSequence<py::tuple, ValueTuple, ConvertAny>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_ANY), py_parse::ConvertSequence<py::list, ValueList, ConvertAny>},
    {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_ANY, mindspore::ops::DT_TUPLE_ANY),
     py_parse::ConvertSingleElementToSequence<ValueTuple, ConvertAny>},
    {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_ANY, mindspore::ops::DT_LIST_ANY),
     py_parse::ConvertSingleElementToSequence<ValueList, ConvertAny>},
    {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_LIST_ANY, mindspore::ops::DT_TUPLE_ANY),
     py_parse::ConvertSequence<py::list, ValueTuple, ConvertAny>},
    {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_ANY, mindspore::ops::DT_LIST_ANY),
     py_parse::ConvertSequence<py::tuple, ValueList, ConvertAny>}};
  return kAnyConverters;
}
}  // namespace

OpDefConvertFunc GetConverterByType(int32_t dtype) {
  const auto &kConverters = py_parse::GetConverters();
  auto it = kConverters.find(dtype);
  if (it == kConverters.end()) {
    const auto &kAnyConverters = GetAnyConverters();
    auto it_any = kAnyConverters.find(dtype);
    if (it_any == kAnyConverters.end()) {
      py_parse::ReportGetConverterError(dtype);
    }
    return it_any->second;
  }
  return it->second;
}

OpDefConvertFunc GetConverterByType(const mindspore::ops::OP_DTYPE &src, const mindspore::ops::OP_DTYPE &dst) {
  return GetConverterByType(py_parse::CombineTypesForTypeCast(src, dst));
}

namespace {
// if the type of op_arg is list[...], convert the value of converted to ValueList
ValuePtr ConvertValueToValueSequence(const ValuePtr &value, ops::OP_DTYPE arg_dtype) {
  auto IsListDtype = [](ops::OP_DTYPE arg_type) {
    constexpr std::pair<ops::OP_DTYPE, ops::OP_DTYPE> list_type_range{ops::OP_DTYPE::DT_LIST_BOOL,
                                                                      ops::OP_DTYPE::DT_LIST_ANY};
    return arg_type <= list_type_range.second && arg_type >= list_type_range.first;
  };
  if (value->isa<ValueTuple>() && IsListDtype(arg_dtype)) {
    return std::make_shared<ValueList>(value->cast<ValueTuplePtr>()->value());
  }
  return value;
}
}  // namespace
ValuePtr DoConvert(const py::object &arg, ops::OP_DTYPE arg_dtype, OpDefConvertFunc converter) {
  MS_EXCEPTION_IF_NULL(converter);
  ValuePtr value = nullptr;
  value = converter(arg);
  if (value != nullptr) {
    return ConvertValueToValueSequence(value, arg_dtype);
  }
  return value;
}
}  // namespace parse
}  // namespace mindspore
