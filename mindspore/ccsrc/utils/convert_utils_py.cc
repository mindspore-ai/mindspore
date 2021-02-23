/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "utils/convert_utils_py.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <list>
#include <utility>
#include <cfloat>

#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/parse_base.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/ms_context.h"

namespace mindspore {
py::object BuiltinsToPyData(const Any &value);
py::object BuiltinsToPyData(const BaseRef &value);
py::object VectorToPyData(const Any &value);
py::object VectorRefToPyData(const VectorRef &value);

py::object TensorToPyData(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->NeedWait()) {
    py::gil_scoped_release release;
    tensor->Wait();
  }
  py::tuple v(1);
  v[0] = tensor;
  return v[0];
}

py::object ScalarPtrToPyData(const ScalarPtr &value) {
  py::int_ int_v;
  py::float_ float_v;
  py::bool_ bool_v;
  TypeId scalar_type = value->type()->type_id();
  switch (scalar_type) {
    case kNumberTypeUInt8:
      MS_LOG(DEBUG) << "uint8";
      int_v = value->cast<UInt8ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeUInt16:
      MS_LOG(DEBUG) << "uint16";
      int_v = value->cast<UInt16ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeUInt32:
      MS_LOG(DEBUG) << "uint32";
      int_v = value->cast<UInt32ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeUInt64:
      MS_LOG(DEBUG) << "uint64";
      int_v = value->cast<UInt64ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeInt8:
      MS_LOG(DEBUG) << "int8";
      int_v = value->cast<Int8ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeInt16:
      MS_LOG(DEBUG) << "int16";
      int_v = value->cast<Int16ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeInt32:
      MS_LOG(DEBUG) << "int32";
      int_v = value->cast<Int32ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeInt64:
      MS_LOG(DEBUG) << "int64";
      int_v = value->cast<Int64ImmPtr>()->value();
      return std::move(int_v);
    case kNumberTypeFloat32:
      MS_LOG(DEBUG) << "float";
      float_v = value->cast<FP32ImmPtr>()->value();
      return std::move(float_v);
    case kNumberTypeFloat64:
      MS_LOG(DEBUG) << "double";
      float_v = value->cast<FP64ImmPtr>()->value();
      return std::move(float_v);
    case kNumberTypeBool:
      MS_LOG(DEBUG) << "bool";
      bool_v = value->cast<BoolImmPtr>()->value();
      return std::move(bool_v);
    default:
      MS_EXCEPTION(TypeError) << "Unsupported scalar converted to py data: " << value->ToString();
  }
}

py::object ValuePtrToPyData(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "value is null";
  }
  py::object ret;
  if (value->isa<Scalar>()) {
    ret = ScalarPtrToPyData(value->cast<ScalarPtr>());
  } else if (value->isa<StringImm>()) {
    MS_LOG(DEBUG) << "String";
    py::str v = value->cast<StringImmPtr>()->value();
    ret = v;
  } else if (value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = value->cast<tensor::TensorPtr>();
    ret = TensorToPyData(tensor_ptr);
  } else if (value->isa<tensor::MetaTensor>()) {
    MS_LOG(DEBUG) << "MetaTensor";
    py::tuple v(1);
    v[0] = value->cast<tensor::MetaTensorPtr>();
    ret = v[0];
  } else if (value->isa<RefKey>()) {
    MS_LOG(DEBUG) << "RefKey";
    py::tuple v(1);
    v[0] = value->cast<RefKeyPtr>();
    ret = v[0];
  } else if (value->isa<ValueSequeue>()) {
    MS_LOG(DEBUG) << "tuple or list";
    auto value_sequeue = value->cast<ValueSequeuePtr>()->value();
    py::tuple ret_sequeue(value_sequeue.size());

    for (size_t i = 0; i < value_sequeue.size(); i++) {
      ret_sequeue[i] = ValuePtrToPyData(value_sequeue[i]);
    }
    if (value->isa<ValueTuple>()) {
      ret = ret_sequeue;
    } else {
      ret = ret_sequeue.cast<py::list>();
    }
  } else if (value->isa<ValueDictionary>()) {
    MS_LOG(DEBUG) << "dict";
    auto value_list = value->cast<ValueDictionaryPtr>()->value();
    py::dict ret_dict;
    for (const auto &v : value_list) {
      ret_dict[py::str(v.first)] = ValuePtrToPyData(v.second);
    }
    ret = ret_dict;
  } else if (value->isa<Ellipsis>()) {
    ret = py::ellipsis();
  } else if (value->isa<ValueSlice>()) {
    auto slice = value->cast<ValueSlicePtr>();
    auto start = ValuePtrToPyData(slice->start());
    auto end = ValuePtrToPyData(slice->stop());
    auto step = ValuePtrToPyData(slice->step());
    ret = parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_CLASS_SLICE, start, end,
                                          step);
  } else if (value->isa<Type>()) {
    py::tuple v(1);
    v[0] = value->cast<TypePtr>();
    ret = v[0];
  } else if (value->isa<AnyValue>() || value->isa<None>() || value->isa<Monad>() || value->isa<FuncGraph>()) {
    // FuncGraph is not used in the backend, return None
    ret = py::none();
  } else {
    MS_LOG(EXCEPTION) << "Unsupported convert value: " << value->ToString() << " to a PyData.";
  }
  return ret;
}

py::object AnyToPyData(const Any &value) {
  py::object ret;
  MS_LOG(DEBUG) << "AnyToPyData " << value.GetString();
  if (value.is<int>() || value.is<float>() || value.is<double>() || value.is<bool>()) {
    ret = BuiltinsToPyData(value);
  } else if (value.is<ValuePtr>()) {
    MS_LOG(DEBUG) << "ValuePtr";
    ValuePtr v = value.cast<ValuePtr>();
    ret = ValuePtrToPyData(v);
  } else if (value.is<tensor::TensorPtr>()) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = value.cast<tensor::TensorPtr>();
    ret = TensorToPyData(tensor_ptr);
  } else if (value.is<py::object>()) {
    MS_LOG(DEBUG) << "py obj";
    ret = value.cast<py::object>();
  } else if (value.is<std::vector<tensor::TensorPtr>>() || value.is<std::vector<Any>>()) {
    ret = VectorToPyData(value);
  } else if (value.is<std::list<Any>>()) {
    MS_LOG(DEBUG) << "list_any";
    auto value_list = value.cast<std::list<Any>>();
    py::list rets = py::list();
    for (auto &v : value_list) {
      rets.append(AnyToPyData(v));
    }
    ret = rets;
  } else if (value.is<std::vector<Any>>()) {
    auto value_list = value.cast<std::vector<Any>>();
    py::tuple rets(value_list.size());
    for (size_t i = 0; i < value_list.size(); i++) {
      rets[i] = AnyToPyData(value_list[i]);
    }
    ret = rets;
  } else if (value.is<TypePtr>()) {
    py::tuple v(1);
    v[0] = value.cast<TypePtr>();
    ret = v[0];
  } else {
    MS_LOG(EXCEPTION) << "value is not support type";
  }
  return ret;
}

py::object BaseRefToPyData(const BaseRef &value) {
  py::object ret;
  MS_LOG(DEBUG) << "BaseRefToPyData " << value.ToString();
  if (utils::isa<int>(value) || utils::isa<float>(value) || utils::isa<double>(value) || utils::isa<bool>(value)) {
    ret = BuiltinsToPyData(value);
  } else if (utils::isa<ValuePtr>(value)) {
    MS_LOG(DEBUG) << "ValuePtr";
    ValuePtr v = utils::cast<ValuePtr>(value);
    ret = ValuePtrToPyData(v);
  } else if (utils::isa<tensor::TensorPtr>(value)) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = utils::cast<tensor::TensorPtr>(value);
    ret = TensorToPyData(tensor_ptr);
  } else if (utils::isa<PyObjectRef>(value)) {
    MS_LOG(DEBUG) << "py obj";
    PyObjectRef py_ref = utils::cast<PyObjectRef>(value);
    ret = py_ref.object_;
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToPyData(vec_ref);
  } else if (utils::isa<TypePtr>(value)) {
    py::tuple v(1);
    v[0] = utils::cast<TypePtr>(value);
    ret = v[0];
  } else {
    MS_LOG(EXCEPTION) << "value is not support type";
  }
  return ret;
}

py::object BuiltinsToPyData(const Any &value) {
  if (value.is<int>()) {
    MS_LOG(DEBUG) << "int";
    py::int_ ret = value.cast<int>();
    return std::move(ret);
  } else if (value.is<float>()) {
    MS_LOG(DEBUG) << "float";
    py::float_ ret = value.cast<float>();
    return std::move(ret);
  } else if (value.is<double>()) {
    MS_LOG(DEBUG) << "double";
    py::float_ ret = value.cast<double>();
    return std::move(ret);
  } else {
    MS_LOG(DEBUG) << "bool";
    py::bool_ ret = value.cast<bool>();
    return std::move(ret);
  }
}

py::object BuiltinsToPyData(const BaseRef &value) {
  if (utils::isa<int>(value)) {
    MS_LOG(DEBUG) << "int";
    py::int_ ret = utils::cast<int>(value);
    return std::move(ret);
  } else if (utils::isa<float>(value)) {
    MS_LOG(DEBUG) << "float";
    py::float_ ret = utils::cast<float>(value);
    return std::move(ret);
  } else if (utils::isa<double>(value)) {
    MS_LOG(DEBUG) << "double";
    py::float_ ret = utils::cast<double>(value);
    return std::move(ret);
  } else {
    MS_LOG(DEBUG) << "bool";
    py::bool_ ret = utils::cast<bool>(value);
    return std::move(ret);
  }
}

py::object VectorToPyData(const Any &value) {
  py::object ret;
  if (value.is<std::vector<tensor::TensorPtr>>()) {
    MS_LOG(DEBUG) << "vector_tensor";
    std::vector<tensor::TensorPtr> outputs;
    outputs = value.cast<std::vector<tensor::TensorPtr>>();
    py::tuple tensor_tuple(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      tensor_tuple[i] = *outputs[i];
    }
    ret = tensor_tuple;
  } else {
    MS_LOG(DEBUG) << "vector_any";
    auto value_list = value.cast<std::vector<Any>>();
    py::tuple any_tuple = py::tuple(value_list.size());
    size_t i = 0;
    for (auto &v : value_list) {
      any_tuple[i] = AnyToPyData(v);
      i++;
    }
    ret = any_tuple;
  }
  return ret;
}

py::object VectorRefToPyData(const VectorRef &value_list) {
  py::object ret;
  MS_LOG(DEBUG) << "vector_ref";
  size_t value_size = value_list.size();
  auto ref_tuple = py::tuple(value_size);
  for (size_t i = 0; i < value_size; i++) {
    ref_tuple[i] = BaseRefToPyData(value_list[i]);
  }
  ret = ref_tuple;
  return ret;
}

void SetValueRange(const AbstractBasePtr &tensor, const py::object &output) {
  if (output.is_none()) {
    return;
  }
  py::object obj_min =
    output.contains(py::str(ATTR_MIN_VALUE)) ? (py::object)output[ATTR_MIN_VALUE] : (py::object)py::none();
  py::object obj_max =
    output.contains(py::str(ATTR_MAX_VALUE)) ? (py::object)output[ATTR_MAX_VALUE] : (py::object)py::none();

  if (!obj_min.is_none() && !obj_max.is_none()) {
    bool converted = true;
    ValuePtr min_value = nullptr;
    ValuePtr max_value = nullptr;
    converted = parse::ConvertData(obj_min, &min_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape min value data failed";
    }
    converted = parse::ConvertData(obj_max, &max_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape max value data failed";
    }
    auto abs_tensor = dyn_cast<abstract::AbstractTensor>(tensor);
    abs_tensor->set_value_range(min_value, max_value);
  }
}

AbstractBasePtr PyList2DynamicShapeTensor(const py::object &shape_obj, const py::object &type_obj,
                                          const py::object &output) {
  AbstractBasePtr tensor = nullptr;
  auto ret_vec = shape_obj.cast<ShapeVector>();
  auto ret_dtype = type_obj.cast<TypePtr>();
  ShapeVector min_shape_vec;
  ShapeVector max_shape_vec;

  if (!output.is_none()) {
    py::object min_shape =
      output.contains(py::str(ATTR_MIN_SHAPE)) ? (py::object)output[ATTR_MIN_SHAPE] : (py::object)py::none();
    py::object max_shape =
      output.contains(py::str(ATTR_MAX_SHAPE)) ? (py::object)output[ATTR_MAX_SHAPE] : (py::object)py::none();
    if (!min_shape.is_none()) {
      min_shape_vec = min_shape.cast<ShapeVector>();
    }
    if (!max_shape.is_none()) {
      max_shape_vec = max_shape.cast<ShapeVector>();
    }
  }

  auto ret_shape = std::make_shared<abstract::Shape>(ret_vec, min_shape_vec, max_shape_vec);
  if (ret_dtype->isa<TensorType>()) {
    auto tensor_type = type_obj.cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, tensor_type->element());
    tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
  } else {
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
    tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
  }

  SetValueRange(tensor, output);
  return tensor;
}

static bool IsMonadType(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    return type->isa<MonadType>();
  }
  return false;
}

static AbstractBasePtr ToMonadAbstract(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    if (type->isa<UMonadType>()) {
      return kUMonad->ToAbstract();
    }
    if (type->isa<IOMonadType>()) {
      return kIOMonad->ToAbstract();
    }
  }
  MS_LOG(EXCEPTION) << "Not a monad type object: " << py::str(type_obj);
}

AbstractBasePtr PyListDtype2AbstractTensor(const py::object &shape_obj, const py::object &type_obj,
                                           const py::object &output) {
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto ret_vec = shape_obj.cast<ShapeVector>();
    auto ret_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (ret_vec.empty() && (!ret_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      return abs_scalar;
    }
    return PyList2DynamicShapeTensor(shape_obj, type_obj, output);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto shape_tuple = shape_obj.cast<py::tuple>();
    auto typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_tuple.size(); ++it) {
      auto tensor_it = PyListDtype2AbstractTensor(shape_tuple[it], typeid_tuple[it]);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto shape_list = shape_obj.cast<py::list>();
    auto typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_list.size(); ++it) {
      auto tensor_it = PyListDtype2AbstractTensor(shape_list[it], typeid_list[it]);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else if (IsMonadType(type_obj)) {
    // Return monad abstract if it is monad type.
    return ToMonadAbstract(type_obj);
  } else {
    // When sparse enabled, the undetermined might be raised and eliminated in opt passes
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
    if (enable_sparse) {
      return std::make_shared<abstract::AbstractUndetermined>();
    }
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << (std::string)py::str(type_obj);
  }
}
bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                       const std::shared_ptr<py::object> &ret_val) {
  if (output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    ValuePtr value = GetValueNode(output);
    *ret_val = ValuePtrToPyData(value);
    return true;
  }

  // Adapter will transform values in __init__() and construct() to parameters, this could cause
  // inputs (a.k.a args in current function) size less than parameters'.
  if (output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    // Find the right parameter as ret_val.
    auto func_graph = output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if ((args.size() + func_graph->hyper_param_count()) != params.size()) {
      MS_LOG(EXCEPTION) << "Input size " << args.size() << " add Parameter count " << func_graph->hyper_param_count()
                        << " not equal to graph input size " << params.size() << ", let graph to be executed.";
    }

    auto it = std::find(params.begin(), params.end(), output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter,  it should be found in graph parameters";
    }
    size_t index = it - params.cbegin();
    if (index >= args.size() + func_graph->hyper_param_count()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size()
                                 << " add Parameter count " << func_graph->hyper_param_count() << ".";
    }
    if (index < args.size()) {
      *ret_val = args[index];
    } else {
      auto param = dyn_cast<Parameter>(params[index]);
      MS_EXCEPTION_IF_NULL(param);
      if (!param->has_default()) {
        MS_LOG(EXCEPTION) << "Can not determine value of Parameter " << index << " (" << param->name() << ")";
      }
      auto tensor = param->default_param();
      *ret_val = py::cast(tensor);
    }
    return true;
  }
  return false;
}
}  // namespace mindspore
