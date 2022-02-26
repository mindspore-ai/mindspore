/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "include/common/utils/convert_utils_py.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <list>
#include <utility>
#include <cfloat>

#include "abstract/abstract_value.h"
#include "abstract/utils.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/parse/resolve.h"
#include "ir/value.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "pybind_api/ir/base_ref_py.h"
#include "ir/dtype/tensor_type.h"
#include "utils/ms_context.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
py::object BuiltinsToPyData(const Any &value);
py::object BuiltinsToPyData(const BaseRef &value);
py::object VectorToPyData(const Any &value);
py::object VectorRefToPyData(const VectorRef &value_list);
py::object VectorRefToPyData(const VectorRef &value_list, const AbstractBasePtr &output);
// Wrap VectorRef to CSRTensor
py::object MakeCSRTensor(const VectorRef &value_list);
py::object MakeCOOTensor(const VectorRef &value_list);
ShapeVector ConvertToShapeVector(const ValuePtr &shape_ptr, const VectorRef &value_list, size_t shape_idx);
py::object CSRTensorToPyData(const tensor::CSRTensorPtr &csr_tensor) {
  auto ref = py::tuple(1);
  ref[0] = csr_tensor;
  return ref[0];
}
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

using ConverterFunction = std::function<py::object(const ValuePtr &value)>;
using ValueNameToConverterVector = std::vector<std::pair<uint32_t, ConverterFunction>>;

// (Value Type Name) -> (Converter Function)
// The converter function is used to convert Value object to Python data object.
static ValueNameToConverterVector value_name_to_converter = {
  // Scalar
  {Scalar::kTypeId, [](const ValuePtr &value) -> py::object { return ScalarPtrToPyData(value->cast<ScalarPtr>()); }},
  // Tensor
  {tensor::Tensor::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto tensor_ptr = value->cast<tensor::TensorPtr>();
     return TensorToPyData(tensor_ptr);
   }},
  // MetaTenser
  {tensor::MetaTensor::kTypeId,
   [](const ValuePtr &value) -> py::object {
     py::tuple tuple_container(1);
     tuple_container[0] = value->cast<tensor::MetaTensorPtr>();
     return tuple_container[0];
   }},
  // CSRTensor
  {tensor::CSRTensor::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto csr_tensor_ptr = value->cast<tensor::CSRTensorPtr>();
     return CSRTensorToPyData(csr_tensor_ptr);
   }},
  // RefKey
  {RefKey::kTypeId,
   [](const ValuePtr &value) -> py::object {
     py::tuple tuple_container(1);
     tuple_container[0] = value->cast<RefKeyPtr>();
     return tuple_container[0];
   }},
  // Type
  {Type::kTypeId,
   [](const ValuePtr &value) -> py::object {
     py::tuple tuple_container(1);
     tuple_container[0] = value->cast<TypePtr>();
     return tuple_container[0];
   }},
  // StringImm
  {StringImm::kTypeId,
   [](const ValuePtr &value) -> py::object {
     py::str res = value->cast<StringImmPtr>()->value();
     return res;
   }},
  // ValueSequence
  {ValueSequence::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto value_sequeue = value->cast<ValueSequencePtr>()->value();
     py::tuple res_sequeue(value_sequeue.size());
     for (size_t i = 0; i < value_sequeue.size(); i++) {
       res_sequeue[i] = ValueToPyData(value_sequeue[i]);
     }
     if (value->isa<ValueTuple>()) {
       return res_sequeue;
     }
     return res_sequeue.cast<py::list>();
   }},
  // ValueDictionary
  {ValueDictionary::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto value_list = value->cast<ValueDictionaryPtr>()->value();
     py::dict res_dict;
     for (const auto &v : value_list) {
       res_dict[py::str(v.first)] = ValueToPyData(v.second);
     }
     return res_dict;
   }},
  // ValueSlice
  {ValueSlice::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto slice = value->cast<ValueSlicePtr>();
     auto start = ValueToPyData(slice->start());
     auto end = ValueToPyData(slice->stop());
     auto step = ValueToPyData(slice->step());
     return python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_CLASS_SLICE, start, end, step);
   }},
  // KeywordArg
  {KeywordArg::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto abs_keyword_arg = value->ToAbstract()->cast<abstract::AbstractKeywordArgPtr>();
     auto key = abs_keyword_arg->get_key();
     auto val = abs_keyword_arg->get_arg()->BuildValue();
     auto py_value = ValueToPyData(val);
     auto kwargs = py::kwargs();
     kwargs[key.c_str()] = py_value;
     return kwargs;
   }},
  // parse::NameSpace
  {parse::NameSpace::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto ns = value->cast<parse::NameSpacePtr>();
     return ns->module_obj();
   }},
  // parse::ClassType
  {parse::ClassType::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto class_type = value->cast<parse::ClassTypePtr>();
     return class_type->obj();
   }},
  // parse::InterpretedObject
  {parse::InterpretedObject::kTypeId,
   [](const ValuePtr &value) -> py::object {
     auto interpreted_object = value->cast<parse::InterpretedObjectPtr>();
     return interpreted_object->obj();
   }},
  // None
  {None::kTypeId, [](const ValuePtr &value) -> py::object { return py::none(); }},
  // AnyValue
  {AnyValue::kTypeId, [](const ValuePtr &value) -> py::object { return py::none(); }},
  // FuncGraph
  {FuncGraph::kTypeId, [](const ValuePtr &value) -> py::object { return py::none(); }},
  // Monad
  {Monad::kTypeId, [](const ValuePtr &value) -> py::object { return py::none(); }},
  // Ellipsis
  {Ellipsis::kTypeId, [](const ValuePtr &value) -> py::object { return py::ellipsis(); }}};

py::object ValueToPyData(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "The `value` should not be null";
  }
  for (auto &iter : value_name_to_converter) {
    if (value->IsFromTypeId(iter.first)) {
      return iter.second(value);
    }
  }
  MS_LOG(EXCEPTION) << "Unsupported to convert " << value->ToString() << "[" << value->type_name() << "] to a PyData";
}

py::object AnyToPyData(const Any &value) {
  py::object ret;
  MS_LOG(DEBUG) << "AnyToPyData " << value.GetString();
  if (value.is<int>() || value.is<float>() || value.is<double>() || value.is<bool>()) {
    ret = BuiltinsToPyData(value);
  } else if (value.is<ValuePtr>()) {
    MS_LOG(DEBUG) << "ValuePtr";
    ValuePtr v = value.cast<ValuePtr>();
    ret = ValueToPyData(v);
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

py::object BaseRefToPyData(const BaseRef &value, const AbstractBasePtr &output) {
  py::object ret;
  // If output value is a tuple, check if abstract is a COOTensor in funcgraph output
  if (utils::isa<VectorRef>(value)) {
    MS_LOG(DEBUG) << "BaseRefToPyData, value is tuple: " << value.ToString();
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToPyData(vec_ref, output);
  } else {
    ret = BaseRefToPyData(value);
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
    ret = ValueToPyData(v);
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

py::object VectorRefToPyData(const VectorRef &value_list, const AbstractBasePtr &output) {
  MS_LOG(DEBUG) << "vector_ref";
  // Current VectorRef reflects a COOTensor type
  if (output->isa<abstract::AbstractCSRTensor>()) {
    return MakeCSRTensor(value_list);
  }
  if (output->isa<abstract::AbstractCOOTensor>()) {
    return MakeCOOTensor(value_list);
  }
  py::object ret;
  size_t value_size = value_list.size();
  auto ref_tuple = py::tuple(value_size);
  abstract::AbstractTuplePtr tuple_output = output->cast<abstract::AbstractTuplePtr>();
  bool is_abstract_tuple = tuple_output != nullptr;
  for (size_t i = 0; i < value_size; i++) {
    if (!is_abstract_tuple || i >= tuple_output->size()) {
      // Fall back to original process
      ref_tuple[i] = BaseRefToPyData(value_list[i]);
    } else {
      ref_tuple[i] = BaseRefToPyData(value_list[i], (*tuple_output)[i]);
    }
  }
  ret = ref_tuple;
  return ret;
}

bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                       const std::shared_ptr<py::object> &ret_val) {
  if (output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    ValuePtr value = GetValueNode(output);
    *ret_val = ValueToPyData(value);
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

ShapeVector ConvertToShapeVector(const ValuePtr &shape_ptr, const VectorRef &value_list, size_t shape_idx) {
  MS_EXCEPTION_IF_NULL(shape_ptr);
  ShapeVector shape;
  ValueTuplePtr shape_tuple = shape_ptr->cast<ValueTuplePtr>();
  if (shape_tuple) {
    for (const auto &v : shape_tuple->value()) {
      MS_EXCEPTION_IF_NULL(v);
      ScalarPtr scalar = v->cast<ScalarPtr>();
      MS_EXCEPTION_IF_NULL(scalar);
      shape.push_back(GetValue<int64_t>(scalar));
    }
  } else {
    auto shape_ref = utils::cast<VectorRef>(value_list[shape_idx]);
    MS_EXCEPTION_IF_NULL(shape_ref);
    for (const auto &v : shape_ref) {
      MS_EXCEPTION_IF_NULL(v);
      auto tensorptr = utils::cast<tensor::TensorPtr>(v);
      MS_EXCEPTION_IF_NULL(tensorptr);
      if (tensorptr->DataDim() != 0) {
        MS_LOG(EXCEPTION) << "Element in COOTensor's shape must be scalar!";
      }
      tensorptr->data_sync(false);
      shape.push_back(*(static_cast<int64_t *>(tensorptr->data_c())));
    }
  }
  return shape;
}

py::object MakeCSRTensor(const VectorRef &value_list) {
  constexpr size_t kCSRTensorInputSize{4};
  if (value_list.size() != kCSRTensorInputSize) {
    MS_LOG(EXCEPTION) << "CSRTensor must have 4 inputs.";
  }
  using TensorPtr = tensor::TensorPtr;
  using CSRTensor = tensor::CSRTensor;
  constexpr size_t kIndptrIdx{0};
  constexpr size_t kIndicesIdx{1};
  constexpr size_t kValuesIdx{2};
  constexpr size_t kShapeIdx{3};
  TensorPtr indptr = utils::cast<TensorPtr>(value_list[kIndptrIdx]);
  TensorPtr indices = utils::cast<TensorPtr>(value_list[kIndicesIdx]);
  TensorPtr values = utils::cast<TensorPtr>(value_list[kValuesIdx]);
  ValuePtr shape_ptr = utils::cast<ValuePtr>(value_list[kShapeIdx]);

  ShapeVector shape = ConvertToShapeVector(shape_ptr, value_list, kShapeIdx);
  auto csr_tensor_ptr = std::make_shared<CSRTensor>(indptr, indices, values, shape);
  return CSRTensorToPyData(csr_tensor_ptr);
}

py::object MakeCOOTensor(const VectorRef &value_list) {
  constexpr size_t kCOOTensorInputSize{3};
  constexpr size_t kIndicesIdx{0};
  constexpr size_t kValuesIdx{1};
  constexpr size_t kShapeIdx{2};
  if (value_list.size() != kCOOTensorInputSize) {
    MS_LOG(EXCEPTION) << "COOTensor must have " << kCOOTensorInputSize << "inputs.";
  }
  tensor::TensorPtr indices = utils::cast<tensor::TensorPtr>(value_list[kIndicesIdx]);
  tensor::TensorPtr values = utils::cast<tensor::TensorPtr>(value_list[kValuesIdx]);
  ValuePtr shape_ptr = utils::cast<ValuePtr>(value_list[kShapeIdx]);

  ShapeVector shape = ConvertToShapeVector(shape_ptr, value_list, kShapeIdx);
  auto ref = py::tuple(1);
  auto coo_tensor_ptr = std::make_shared<tensor::COOTensor>(indices, values, shape);
  ref[0] = coo_tensor_ptr;
  return ref[0];
}
}  // namespace mindspore
