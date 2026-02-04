/**
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

#include "include/frontend/jit/ps/static_analysis/py_infer_convert.h"

#include <string>
#include <vector>
#include <utility>

#include "utils/flags.h"
#include "utils/log_adapter.h"
#include "ir/core_ops_primitive.h"
#include "abstract/abstract_function.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/static_analysis/auto_monad.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"

namespace mindspore {
namespace abstract {
namespace {
py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value = false);

py::object BuildPyObject(const ValuePtr &value_ptr) {
  if (value_ptr == nullptr) {
    return py::none();
  } else {
    return ValueToPyData(value_ptr);
  }
}

py::object AbstractTupleValueToPython(const AbstractTuple *tuple_abs) {
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if (tuple_abs->dynamic_len()) {
    return py::none();
  }
  const auto &elements = tuple_abs->elements();
  size_t len = elements.size();
  py::tuple value_tuple(len);
  for (size_t i = 0; i < len; ++i) {
    value_tuple[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_tuple;
}

py::dict AbstractTupleToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_tuple = dyn_cast_ptr<AbstractTuple>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_tuple);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractTupleValueToPython(arg_tuple);
    return dic;
  }
  if (arg_tuple->dynamic_len()) {
    dic[ATTR_VALUE] = py::none();
    dic[ATTR_SHAPE] = ShapeVector{abstract::Shape::kShapeDimAny};
    dic[ATTR_DTYPE] = arg_tuple->BuildType();
    return dic;
  }
  size_t len = arg_tuple->size();
  py::tuple shape_tuple(len);
  py::tuple dtype_tuple(len);
  py::tuple value_tuple(len);
  std::vector<py::dict> res;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_tuple->elements()[i]);
    res.push_back(out);
    shape_tuple[i] = out[ATTR_SHAPE];
    dtype_tuple[i] = out[ATTR_DTYPE];
    value_tuple[i] = out[ATTR_VALUE];
  }
  dic[ATTR_SHAPE] = shape_tuple;
  dic[ATTR_DTYPE] = dtype_tuple;
  dic[ATTR_VALUE] = value_tuple;

  return dic;
}

py::dict AbstractDictionaryToPython(const AbstractBasePtr &abs_base) {
  auto arg_dict = dyn_cast_ptr<AbstractDictionary>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_dict);

  size_t len = arg_dict->size();
  const auto &arg_dict_elements = arg_dict->elements();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::dict value_dict = py::dict();

  for (size_t i = 0; i < len; ++i) {
    auto cur_attr = arg_dict_elements[i];
    auto cur_key = cur_attr.first;
    auto cur_value = cur_attr.second;

    py::dict cur_value_out = ConvertAbstractToPython(cur_value);
    shape_list[i] = cur_value_out[ATTR_SHAPE];
    dtype_list[i] = cur_value_out[ATTR_DTYPE];
    MS_EXCEPTION_IF_NULL(cur_key);
    value_dict[ValueToPyData(cur_key->BuildValue())] = cur_value_out[ATTR_VALUE];
  }

  py::dict dic = py::dict();
  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  MS_EXCEPTION_IF_NULL(arg_dict->BuildValue());
  dic[ATTR_VALUE] = value_dict;
  return dic;
}

py::object AbstractKWArgsToPython(const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto abs_keyword_arg = abs_base->cast_ptr<abstract::AbstractKeywordArg>();
  MS_EXCEPTION_IF_NULL(abs_keyword_arg);
  auto args_abs = abs_keyword_arg->get_arg();
  auto args_obj = BuildPyObject(args_abs->BuildValue());
  // if the args is none but the type is not none means the input is a variable.
  if (!args_abs->isa<AbstractNone>() && py::isinstance<py::none>(args_obj)) {
    return py::none();
  }
  return BuildPyObject(abs_base->BuildValue());
}

py::object AbstractListValueToPython(const AbstractList *list_abs) {
  MS_EXCEPTION_IF_NULL(list_abs);
  if (list_abs->dynamic_len()) {
    return py::none();
  }
  const auto &elements = list_abs->elements();
  size_t len = elements.size();
  py::list value_list(len);
  for (size_t i = 0; i < len; ++i) {
    value_list[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_list;
}

py::dict AbstractListToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_list = dyn_cast_ptr<AbstractList>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_list);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractListValueToPython(arg_list);
    return dic;
  }
  if (arg_list->dynamic_len()) {
    auto elem_out = ConvertAbstractToPython(arg_list->dynamic_len_element_abs());
    dic[ATTR_VALUE] = py::none();
    dic[ATTR_SHAPE] = elem_out[ATTR_SHAPE];
    dic[ATTR_DTYPE] = elem_out[ATTR_DTYPE];
    return dic;
  }
  size_t len = arg_list->size();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::list value_list(len);
  std::vector<py::dict> res;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
    res.push_back(out);
    shape_list[i] = out[ATTR_SHAPE];
    dtype_list[i] = out[ATTR_DTYPE];
    value_list[i] = out[ATTR_VALUE];
  }

  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  dic[ATTR_VALUE] = value_list;
  return dic;
}

void ConvertAbstractTensorToPython(const AbstractBasePtr &abs_base, bool only_convert_value, py::dict *dic) {
  auto arg_tensor = dyn_cast_ptr<AbstractTensor>(abs_base);
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(arg_tensor);
  if (only_convert_value) {
    (*dic)[ATTR_VALUE] = BuildPyObject(arg_tensor->BuildValue());
    return;
  }
  MS_EXCEPTION_IF_NULL(arg_tensor->shape());
  (*dic)[ATTR_SHAPE] = arg_tensor->shape()->shape();

  (*dic)[ATTR_DTYPE] = arg_tensor->BuildType();
  (*dic)[ATTR_VALUE] = BuildPyObject(arg_tensor->BuildValue());
}

py::object GetPyObjForPrimitiveAbstract(const PrimitiveAbstractClosurePtr &prim_abs) {
  MS_EXCEPTION_IF_NULL(prim_abs);
  auto prim = prim_abs->BuildValue();
  if (prim == nullptr) {
    return py::none();
  }
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    auto do_sig_prim = prim->cast_ptr<prim::DoSignaturePrimitive>();
    auto value = do_sig_prim->function();
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<PrimitivePy>()) {
      return py::none();
    }
    auto prim_py = value->cast_ptr<PrimitivePy>();
    MS_EXCEPTION_IF_NULL(prim_py);
    return prim_py->GetPyObj();
  }
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast_ptr<PrimitivePy>();
    return prim_py->GetPyObj();
  }
  return py::none();
}

void ConvertAbstractFunctionToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(abs_base);
  (*dic)[ATTR_SHAPE] = py::none();
  (*dic)[ATTR_DTYPE] = abs_base->BuildType();
  (*dic)[ATTR_VALUE] = py::none();
  if (abs_base->isa<PartialAbstractClosure>()) {
    auto partial_abs = abs_base->cast<PartialAbstractClosurePtr>();
    AbstractBasePtrList args = partial_abs->args();
    if (!args.empty()) {
      auto value = args[0]->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      auto value_obj = value->cast_ptr<parse::ClassType>();
      if (value_obj != nullptr) {
        (*dic)[ATTR_DTYPE] = std::make_shared<TypeType>();
        (*dic)[ATTR_VALUE] = value_obj->obj();
      }
    }
  }
  if (abs_base->isa<PrimitiveAbstractClosure>()) {
    (*dic)[ATTR_VALUE] = GetPyObjForPrimitiveAbstract(abs_base->cast<PrimitiveAbstractClosurePtr>());
  }
}

void UnknownAbstract(const AbstractBasePtr &abs_base) {
  auto value = abs_base->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  if ((*value == *kValueAny)) {
    auto value_desc = abs_base->value_desc();
    MS_EXCEPTION(TypeError) << "Unsupported parameter " << (value_desc.empty() ? "type" : value_desc)
                            << " for python primitive." << abs_base->ToString();
  }
  MS_EXCEPTION(TypeError) << "Unsupported parameter type for python primitive, the parameter value is "
                          << value->ToString();
}

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto dic = py::dict();
  if (abs_base->isa<AbstractTensor>()) {
    ConvertAbstractTensorToPython(abs_base, only_convert_value, &dic);
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>()) {
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractTuple>()) {
    return AbstractTupleToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractList>()) {
    return AbstractListToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractDictionary>()) {
    return AbstractDictionaryToPython(abs_base);
  } else if (abs_base->isa<AbstractSlice>()) {
    auto arg_slice = dyn_cast_ptr<AbstractSlice>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_slice->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg_slice->BuildValue());
  } else if (abs_base->isa<AbstractRowTensor>()) {
    auto arg = dyn_cast_ptr<AbstractRowTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractCOOTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCOOTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           MS_EXCEPTION_IF_NULL(e);
                           MS_EXCEPTION_IF_NULL(e->cast_ptr<AbstractScalar>());
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractCSRTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCSRTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           MS_EXCEPTION_IF_NULL(e);
                           MS_EXCEPTION_IF_NULL(e->cast_ptr<AbstractScalar>());
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractEllipsis>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::ellipsis();
    dic[ATTR_VALUE] = py::ellipsis();
  } else if (abs_base->isa<AbstractNone>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::none();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractFunction>()) {
    ConvertAbstractFunctionToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractClass>()) {
    auto arg_class = dyn_cast_ptr<AbstractClass>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_class->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg_class->BuildValue());
  } else if (abs_base->isa<AbstractUndetermined>()) {
    auto arg = dyn_cast_ptr<AbstractUndetermined>(abs_base);
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractMonad>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractKeywordArg>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = AbstractKWArgsToPython(abs_base);
  } else {
    UnknownAbstract(abs_base);
  }
  return dic;
}

void CheckCustomPrimOutputInferResult(const PrimitivePtr &prim, const AbstractBasePtr &res_spec) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(res_spec);
  const string kOutputNum = "output_num";
  if (prim->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    auto output_num_value = prim->GetAttr(kOutputNum);
    if (output_num_value == nullptr) {
      MS_LOG(DEBUG) << "The output num may no need to check";
      return;
    }
    int64_t output_num = GetValue<int64_t>(output_num_value);
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]: " << output_num << ", not matches the infer result "
                        << res_spec->ToString();
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast_ptr<AbstractTuple>()->size() != LongToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]: " << output_num << ", not matches the infer result "
                        << res_spec->ToString();
    }
  }
}

static bool IsMonadType(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    return type->isa<MonadType>();
  }
  return false;
}

AbstractBasePtr ToMonadAbstract(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    if (!type->isa<MonadType>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Not a monad type object: " << py::str(type_obj);
    }
    return abstract::MakeMonadAbstract(type->cast<MonadTypePtr>());
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Not a type object: " << py::str(type_obj);
}

py::object GetPyAbsItemOfTupleOut(const py::object &output, const size_t index) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  auto out_item = py::dict();
  auto shape_tuple = shape_obj.cast<py::tuple>();
  auto typeid_tuple = type_obj.cast<py::tuple>();
  out_item[ATTR_DTYPE] = typeid_tuple[index];
  out_item[ATTR_SHAPE] = shape_tuple[index];
  out_item[ATTR_VALUE] = py::none();
  return out_item;
}

AbstractBasePtr MakePyInferRes2AbstractTensor(const py::object &shape_obj, const py::object &type_obj) {
  auto res_vec = shape_obj.cast<ShapeVector>();
  auto res_dtype = type_obj.cast<TypePtr>();

  auto res_shape = std::make_shared<abstract::Shape>(res_vec);
  AbstractBasePtr tensor = MakeAbstractTensor(res_shape, res_dtype);
  return tensor;
}

AbstractBasePtr MakePyInferRes2Abstract(const py::object &output) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto res_vec = shape_obj.cast<ShapeVector>();
    auto res_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(res_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (res_vec.empty() && (!res_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kValueAny, res_dtype);
      return abs_scalar;
    }
    return MakePyInferRes2AbstractTensor(shape_obj, type_obj);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_tuple.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_list.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
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
    MS_LOG(INTERNAL_EXCEPTION) << "Python evaluator return invalid shape or type. " << py::str(type_obj);
  }
}
}  // namespace

py::tuple PreparePyInputs(const AbstractBasePtrList &args) {
  // The monad parameter is defined at the end of the parameter and needs to be ignored
  std::size_t args_size = args.size() - GetAbstractMonadNum(args);
  py::tuple py_args(args_size);
  for (size_t i = 0; i < args_size; i++) {
    py_args[i] = ConvertAbstractToPython(args[i]);
  }
  return py_args;
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  if (output[ATTR_VALUE].is_none()) {
    return MakePyInferRes2Abstract(output);
  }

  // Convert pyobject to Value, then to AbstractValue
  auto out_dtype = output[ATTR_DTYPE];
  TypePtr dtype = py::isinstance<Type>(out_dtype) ? out_dtype.cast<TypePtr>() : nullptr;
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(output[ATTR_VALUE], &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(INTERNAL_EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
  }
  CheckCustomPrimOutputInferResult(prim_py, res_spec);
  return res_spec;
}
}  // namespace abstract
}  // namespace mindspore
