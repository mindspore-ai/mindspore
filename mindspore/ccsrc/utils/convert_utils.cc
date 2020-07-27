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

#include "utils/convert_utils.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <list>
#include <utility>
#include <cfloat>

#include "pybind11/pybind11.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/parse_base.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/param_value.h"
#include "utils/base_ref_extends.h"
#include "utils/ms_context.h"

namespace mindspore {
py::object BuiltinsToPyData(const Any &value);
py::object BuiltinsToPyData(const BaseRef &value);
py::object VectorToPyData(const Any &value);
py::object VectorRefToPyData(const VectorRef &value);

py::object ValuePtrToPyData(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "value is null";
  }
  py::object ret;
  if (value->isa<Int32Imm>()) {
    MS_LOG(DEBUG) << "int";
    py::int_ v = value->cast<Int32ImmPtr>()->value();
    ret = v;
  } else if (value->isa<Int64Imm>()) {
    MS_LOG(DEBUG) << "int64";
    py::int_ v = value->cast<Int64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<UInt64Imm>()) {
    MS_LOG(DEBUG) << "uint64";
    py::int_ v = value->cast<UInt64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<BoolImm>()) {
    MS_LOG(DEBUG) << "bool";
    py::bool_ v = value->cast<BoolImmPtr>()->value();
    ret = v;
  } else if (value->isa<FP64Imm>()) {
    MS_LOG(DEBUG) << "double";
    py::float_ v = value->cast<FP64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<FP32Imm>()) {
    MS_LOG(DEBUG) << "float";
    py::float_ v = value->cast<FP32ImmPtr>()->value();
    ret = v;
  } else if (value->isa<StringImm>()) {
    MS_LOG(DEBUG) << "String";
    py::str v = value->cast<StringImmPtr>()->value();
    ret = v;
  } else if (value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "tensor";
    py::tuple v(1);
    v[0] = value->cast<tensor::TensorPtr>();
    ret = v[0];
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
  } else if (value->isa<ValueTuple>()) {
    MS_LOG(DEBUG) << "tuple";
    auto value_tuple = value->cast<ValueTuplePtr>()->value();
    py::tuple rets(value_tuple.size());

    size_t i = 0;
    for (auto &v : value_tuple) {
      rets[i] = ValuePtrToPyData(v);
      i++;
    }
    ret = rets;
  } else if (value->isa<ValueList>()) {
    MS_LOG(DEBUG) << "list";
    auto value_list = value->cast<ValueListPtr>()->value();
    py::list rets(value_list.size());

    size_t i = 0;
    for (auto &v : value_list) {
      rets[i] = ValuePtrToPyData(v);
      i++;
    }
    ret = rets;
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
  } else if (value->isa<AnyValue>()) {
    ret = py::none();
  } else if (value->isa<None>()) {
    ret = py::none();
  } else {
    MS_LOG(INFO) << "Unsupported convert value: " << value->ToString() << " to a PyData.";
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
    py::tuple v(1);
    v[0] = value.cast<tensor::TensorPtr>();
    ret = v[0];
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
    py::tuple v(1);
    v[0] = utils::cast<tensor::TensorPtr>(value);
    ret = v[0];
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

bool ValueToBool(const ValuePtr &v, bool *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<BoolImm>()) {
    *value = v->cast<BoolImmPtr>()->value();
  } else if (v->isa<Int32Imm>()) {
    *value = v->cast<Int32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<UInt32Imm>()) {
    *value = v->cast<UInt32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<FP32Imm>()) {
    *value = v->cast<FP32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<FP64Imm>()) {
    *value = v->cast<FP64ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    (void)tensor->data_sync();
    bool *tensor_data = static_cast<bool *>(tensor->data_c());
    // maybe need to support if tensor is a bool array
    auto vb = tensor_data[0];
    *value = vb;
  } else {
    MS_LOG(WARNING) << "value is not supported to cast to be bool";
    return false;
  }
  return true;
}

bool BaseRefToInt(const ValuePtr &v, int *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    (void)tensor->data_sync();
    int *tensor_data = static_cast<int *>(tensor->data_c());
    auto vb = tensor_data[0];
    *value = vb;
    return true;
  }
  MS_LOG(ERROR) << "Index must be tensor type.";
  return false;
}

bool BaseRefToBool(const BaseRef &v, bool *value) {
  if (utils::isa<ValuePtr>(v)) {
    return ValueToBool(utils::cast<ValuePtr>(v), value);
  } else if (utils::isa<bool>(v)) {
    auto vb = utils::cast<bool>(v);
    if (vb == true) {
      *value = true;
    } else {
      *value = false;
    }
  } else if (utils::isa<int>(v)) {
    auto vb = utils::cast<int>(v);
    if (vb == 0) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<unsigned int>(v)) {
    auto vb = utils::cast<unsigned int>(v);
    if (vb == 0) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<float>(v)) {
    auto vb = utils::cast<float>(v);
    if (vb >= -FLT_EPSILON && vb <= FLT_EPSILON) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<double>(v)) {
    auto vb = utils::cast<double>(v);
    if (vb >= -DBL_EPSILON && vb <= DBL_EPSILON) {
      *value = false;
    } else {
      *value = true;
    }
  } else {
    MS_LOG(DEBUG) << "value is not supported to cast to be bool";
    return false;
  }
  return true;
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

AbstractBasePtr PyListDtype2AbstractTensor(const py::object &shape_obj, const py::object &type_obj,
                                           const py::object &min_shape, const py::object &max_shape) {
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto ret_vec = shape_obj.cast<std::vector<int>>();
    auto ret_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (ret_vec.empty() && (!ret_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      return abs_scalar;
    }
    AbstractBasePtr tensor = nullptr;
    std::vector<int> min_shape_vec;
    std::vector<int> max_shape_vec;
    if (!min_shape.is_none()) {
      min_shape_vec = min_shape.cast<std::vector<int>>();
    }
    if (!max_shape.is_none()) {
      max_shape_vec = max_shape.cast<std::vector<int>>();
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
    return tensor;
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    py::tuple shape_tuple = shape_obj.cast<py::tuple>();
    py::tuple typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_tuple.size(); ++it) {
      auto tensor_it = PyListDtype2AbstractTensor(shape_tuple[it], typeid_tuple[it]);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else {
    // When sparse enabled, the undetermined might be raised and eliminated in opt passes
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->enable_sparse();
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
    if (args.empty()) {
      MS_LOG(EXCEPTION) << "Inputs size is 0, let graph to be executed.";
    }
    // Find the right parameter as ret_val.
    auto func_graph = output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if (params.empty()) {
      MS_EXCEPTION(UnknownError) << "Graph's parameters size is 0";
    }
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

// Isomorphism
static bool SameNodeShallow(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
                            NodeMapEquiv *const equiv_node) {
  if (equiv_node == nullptr) {
    MS_LOG(ERROR) << "Invalid equiv_node";
    return false;
  }
  if (equiv_node->count(node1) > 0 && (*equiv_node)[node1] == node2) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node1) && IsValueNode<FuncGraph>(node2)) {
    return Isomorphic(GetValueNode<FuncGraphPtr>(node1), GetValueNode<FuncGraphPtr>(node2), equiv_func_graph,
                      equiv_node);
  }
  if (node1->isa<ValueNode>() && node2->isa<ValueNode>()) {
    auto a1 = GetValueNode(node1);
    auto a2 = GetValueNode(node2);
    if (a1->isa<Primitive>() && a2->isa<Primitive>()) {
      return a1->cast<PrimitivePtr>()->name() == a2->cast<PrimitivePtr>()->name();
    } else if (a1->isa<tensor::Tensor>() && a2->isa<tensor::Tensor>()) {
      return a1->cast<tensor::TensorPtr>()->ValueEqual(*(a2->cast<tensor::TensorPtr>()));
    } else {
      return *a1 == *a2;
    }
  }
  if (node1->isa<Parameter>() && node2->isa<Parameter>()) {
    auto para1 = node1->cast<ParameterPtr>();
    auto para2 = node2->cast<ParameterPtr>();
    if (para1->name() == para2->name()) {
      return true;
    }
    MS_LOG(DEBUG) << "two parameters are not equal.";
    return false;
  }
  MS_LOG(ERROR) << "type error";
  return false;
}

static bool SameNode(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
                     NodeMapEquiv *const equiv_node) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  if (node1->isa<CNode>() && node2->isa<CNode>()) {
    auto &inputs1 = node1->cast<CNodePtr>()->inputs();
    auto &inputs2 = node2->cast<CNodePtr>()->inputs();
    for (std::size_t i = 0; i < inputs1.size(); ++i) {
      if (!SameNodeShallow(inputs1[i], inputs2[i], equiv_func_graph, equiv_node)) {
        return false;
      }
    }
    return true;
  }
  return SameNodeShallow(node1, node2, equiv_func_graph, equiv_node);
}

static bool SameSubgraph(AnfNodePtr root1, AnfNodePtr root2, FuncGraphPairMapEquiv *equiv_func_graph,
                         NodeMapEquiv *const equiv_node) {
  std::unordered_set<AnfNodePtr> done;
  std::stack<std::pair<AnfNodePtr, AnfNodePtr>> todo;

  todo.push(std::make_pair(root1, root2));
  while (todo.size() > 0) {
    AnfNodePtr node1 = todo.top().first;
    if (done.count(node1) > 0) {
      todo.pop();
      continue;
    }
    AnfNodePtr node2 = todo.top().second;

    bool condition = false;
    std::vector<AnfNodePtr> s1 = SuccIncoming(node1);
    std::vector<AnfNodePtr> s2 = SuccIncoming(node2);

    if (s1.size() != s2.size()) {
      return false;
    }
    for (std::size_t i = 0; i < s1.size(); ++i) {
      if (done.count(s1[i]) == 0) {
        todo.push(std::make_pair(s1[i], s2[i]));
        condition = true;
      }
    }
    if (condition) {
      continue;
    }
    (void)done.insert(node1);

    auto res = SameNode(node1, node2, equiv_func_graph, equiv_node);
    if (res) {
      (*equiv_node)[node1] = node2;
    } else {
      return false;
    }
    todo.pop();
  }
  return true;
}

bool Isomorphic(FuncGraphPtr fg1, FuncGraphPtr fg2, FuncGraphPairMapEquiv *equiv_func_graph,
                NodeMapEquiv *const equiv_node) {
  auto fg1_fg2 = std::make_pair(fg1, fg2);
  if (equiv_func_graph == nullptr) {
    MS_LOG(ERROR) << "equiv_func_graph not init";
    return false;
  }
  if (equiv_func_graph->find(fg1_fg2) != equiv_func_graph->end()) {
    return (*equiv_func_graph)[fg1_fg2] != kNotEquiv;
  }
  if (fg1 == nullptr || fg2 == nullptr) {
    MS_LOG(ERROR) << "Invalid function graph";
    return false;
  }
  if (fg1->parameters().size() != fg2->parameters().size()) {
    MS_LOG(DEBUG) << "parameters size not match";
    return false;
  }
  if (equiv_node != nullptr) {
    for (std::size_t i = 0; i < fg1->parameters().size(); ++i) {
      (*equiv_node)[fg1->parameters()[i]] = fg2->parameters()[i];
    }
    (*equiv_func_graph)[fg1_fg2] = kPending;
    auto result = SameSubgraph(fg1->get_return(), fg2->get_return(), equiv_func_graph, equiv_node);
    (*equiv_func_graph)[fg1_fg2] = EquivState(result);
    return result;
  }

  MS_LOG(ERROR) << "equiv_node not init";
  return false;
}

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  tensor::TensorPtr tensor = nullptr;
  if (scalar->isa<FloatImm>()) {
    tensor = std::make_shared<tensor::Tensor>(static_cast<double>(GetValue<float>(scalar)), kFloat32);
  } else if (scalar->isa<IntergerImm>()) {
    tensor = std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int>(scalar)), kInt32);
  } else if (scalar->isa<BoolImm>()) {
    const int64_t bool_value = GetValue<bool>(scalar) ? 1 : 0;
    tensor = std::make_shared<tensor::Tensor>(bool_value, kBool);
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_LOG(EXCEPTION) << "Invalid scalar type: " << type_str;
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}

void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(tensors);
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        tensors->push_back(tensor);
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    tensor::TensorPtr tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensors->push_back(tensor);
  }
}
}  // namespace mindspore
