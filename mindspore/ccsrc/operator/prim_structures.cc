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

#include "pipeline/static_analysis/prim.h"
#include "pipeline/static_analysis/utils.h"
#include "pipeline/static_analysis/param_validator.h"
#include "operator/ops.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace abstract {

AbstractBasePtr InferImplStringEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " requires 2 parameters are string, but got param0: " << value_x->ToString()
                      << ", param1: " << value_y->ToString();
  }

  bool ret = (value_x->cast<StringImmPtr>()->value() == value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
}

AbstractBasePtr InferImplStringConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " requires 2 parameters are string, but got param0: " << value_x->ToString()
                      << ", param1: " << value_y->ToString();
  }

  std::string ret = (value_x->cast<StringImmPtr>()->value() + value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
}

AbstractBasePtr InferImplMakeTuple(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  return std::make_shared<AbstractTuple>(args_spec_list);
}

AbstractBasePtr InferImplMakeList(const AnalysisEnginePtr &, const PrimitivePtr &,
                                  const AbstractBasePtrList &args_spec_list) {
  return std::make_shared<AbstractList>(args_spec_list);
}

AbstractBasePtr InferImplMakeDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr keys = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTuplePtr values = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  size_t keys_size = keys->size();
  if (values->size() != keys_size) {
    MS_LOG(EXCEPTION) << op_name << " evaluator keys' size is not equal with values' size";
  }

  std::vector<AbstractAttribute> key_value;
  AbstractScalarPtr key;
  AbstractBasePtrList key_list = keys->elements();
  AbstractBasePtrList value_list = values->elements();
  for (size_t index = 0; index < keys_size; index++) {
    key = CheckArg<AbstractScalar>(op_name + "key", key_list, index);
    ValuePtr keyPtr = key->BuildValue();
    MS_EXCEPTION_IF_NULL(keyPtr);
    if (!keyPtr->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << op_name << " evaluator keys should be string, but got " << keyPtr->ToString();
    }
    std::string key_string = GetValue<std::string>(keyPtr);
    key_value.emplace_back(key_string, value_list[index]);
  }
  return std::make_shared<AbstractDictionary>(key_value);
}

AbstractBasePtr InferImplMakeKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a string and an object of a subclass of AbstractBase.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);

  ValuePtr keyPtr = key->BuildValue();
  if (!keyPtr->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << keyPtr->ToString();
  }
  std::string key_string = GetValue<std::string>(keyPtr);
  return std::make_shared<AbstractKeywordArg>(key_string, args_spec_list[1]);
}

AbstractBasePtr InferImplExtractKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: a string and a keyword.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractKeywordArgPtr kwarg = CheckArg<AbstractKeywordArg>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << key_value->ToString();
  }
  std::string key_input = GetValue<std::string>(key_value);
  std::string key_actual = kwarg->get_key();
  if (key_actual != key_input) {
    MS_LOG(EXCEPTION) << op_name << " evaluator input key should be same as AbstractKeywordArg' key, but input is "
                      << key_input << ", AbstractKeywordArg' key is " << key_actual;
  }
  return kwarg->get_arg();
}

AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: three scalars whose value is an int32 number.
  CheckArgsSize(primitive->name(), args_spec_list, 3);
  size_t args_size = args_spec_list.size();
  for (size_t index = 0; index < args_size; index++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[index]);
    if (!args_spec_list[index]->isa<AbstractScalar>() && !args_spec_list[index]->isa<AbstractNone>()) {
      MS_LOG(EXCEPTION) << "MakeSlice eval " << index << " parameter is neither AbstractScalar nor AbstractNone.";
    }
    if (args_spec_list[index]->isa<AbstractScalar>() &&
        !dyn_cast<AbstractScalar>(args_spec_list[index])->BuildValue()->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "MakeSlice eval " << index << " parameter is an AbstractScalar, but is not an int32 number.";
    }
  }
  // Slice: start, end, step
  return std::make_shared<AbstractSlice>(args_spec_list[0], args_spec_list[1], args_spec_list[2]);
}

// Eval the return type of make_record
AbstractBasePtr InferImplMakeRecord(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: at lease two objects of a subclass of AbstractBase.
  if (args_spec_list.size() < 2) {
    MS_LOG(EXCEPTION) << "Typeof evaluator requires more than 1 parameter, while the input size is "
                      << args_spec_list.size() << ".";
  }

  // args_spec_list[0] maybe AbstractScalarPtr or AbstractTypePtr
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kMetaTypeTypeType) {
    MS_LOG(EXCEPTION) << "Can not make type(" << type->ToString() << ")not TypeType";
  }

  ValuePtr value_track = args_spec_list[0]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  TypePtr type_ptr = value_track->cast<TypePtr>();
  if (type_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Value type error, not Me type:" << value_track->ToString();
  }

  auto cls = dyn_cast<Class>(type_ptr);
  MS_EXCEPTION_IF_NULL(cls);
  ClassAttrVector attributes = cls->GetAttributes();
  CheckArgsSize(primitive->name(), args_spec_list, attributes.size() + 1);

  std::vector<AbstractAttribute> abs_attributes;
  for (size_t i = 0; i < attributes.size(); i++) {
    AbstractAttribute elem(attributes[i].first, args_spec_list[i + 1]);
    abs_attributes.push_back(elem);
  }

  return std::make_shared<AbstractClass>(cls->tag(), abs_attributes, cls->methods());
}

template <typename T>
AbstractBasePtr InferTupleOrListGetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list and a scalar whose value is an int32 number.
  CheckArgsSize(op_name, args_spec_list, 2);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  if (!index_value->isa<Int32Imm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator index should be an int32 number, but got " << index_value->ToString();
  }
  int idx_v = GetValue<int>(index_value);
  std::size_t nelems = queue->elements().size();
  if (idx_v >= SizeToInt(nelems) || idx_v < -SizeToInt(nelems)) {
    MS_LOG(EXCEPTION) << op_name << " evaluator index should be in range[-" << SizeToInt(nelems) << ", "
                      << SizeToInt(nelems) << "), but got " << idx_v << ".";
  }

  std::size_t uidx_v = 0;
  if (idx_v >= 0) {
    uidx_v = IntToSize(idx_v);
  } else {
    uidx_v = IntToSize(idx_v + SizeToInt(nelems));
  }
  return queue->elements()[uidx_v];
}

template <typename T>
AbstractBasePtr InferTupleOrListSetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list, a scalar whose value is an int32 number and an object of a subclass of AbstractBase.
  CheckArgsSize(op_name, args_spec_list, 3);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  if (!index_value->isa<Int32Imm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator index should be an int32 number, but got " << index_value->ToString();
  }
  int idx_v = GetValue<int>(index_value);
  if (idx_v < 0) {
    MS_LOG(EXCEPTION) << "The index of " << typeid(T).name() << " should be positive number, but got " << idx_v << ".";
  }

  size_t uidx_v = IntToSize(idx_v);
  AbstractBasePtrList elements = queue->elements();
  std::size_t nelems = elements.size();
  if (uidx_v >= nelems) {
    MS_LOG(EXCEPTION) << op_name << " evaluator the index: " << uidx_v << " to set out of range: " << nelems - 1 << ".";
  }
  elements[uidx_v] = args_spec_list[2];
  return std::make_shared<T>(elements);
}

AbstractBasePtr InferImplTupleGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListGetItem<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListGetItem<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplTupleSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListSetItem<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListSetItem<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplDictGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict and a scalar whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << key_value->ToString();
  }
  auto key_str = GetValue<std::string>(key_value);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [key_str](const AbstractAttribute &item) { return item.first == key_str; });

  if (it == dict_elems.end()) {
    MS_LOG(EXCEPTION) << "The key " << key_str << " does not exist in the dict:" << args_spec_list[0]->ToString();
  }
  return it->second;
}

AbstractBasePtr InferImplDictSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict and a scalar whose value is a string and an object of a subclass of AbstractBase.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << key_value->ToString();
  }
  std::string key_str = GetValue<std::string>(key_value);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [key_str](AbstractAttribute &item) { return item.first == key_str; });

  MS_EXCEPTION_IF_NULL(args_spec_list[2]);
  auto new_ele = std::make_pair(key_str, args_spec_list[2]);
  if (it != dict_elems.end()) {
    int index = it - dict_elems.begin();
    dict_elems[IntToSize(index)] = new_ele;
  } else {
    dict_elems.push_back(new_ele);
  }
  return std::make_shared<AbstractDictionary>(dict_elems);
}

AbstractBasePtr InferImplListAppend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: a list and an object of a subclass of AbstractBase.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractListPtr list = CheckArg<AbstractList>(op_name, args_spec_list, 0);
  (void)AbstractJoin(list->elements());
  return list;
}

template <typename T>
AbstractBasePtr InferTupleOrListOrDictLen(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list or dict.
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<T>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractScalar>(SizeToInt(arg->size()));
}

AbstractBasePtr InferImplTupleLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplDictLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractDictionary>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplArrayLen(const AnalysisEnginePtr &, const PrimitivePtr &,
                                  const AbstractBasePtrList &args_spec_list) {
  return std::make_shared<AbstractScalar>(kAnyValue, kInt32);
}

AbstractBasePtr InferImplListMap(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: fn, list1, list2, ...
  MS_EXCEPTION_IF_NULL(engine);
  if (args_spec_list.size() <= 1) {
    MS_LOG(EXCEPTION) << "List_map requires at least 1 list. while the input size is  " << args_spec_list.size() << ".";
  }
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(primitive->name(), args_spec_list, 0);
  // check args from 1.
  CheckArgsSpec<AbstractList>(AbstractBasePtrList(args_spec_list.begin() + 1, args_spec_list.end()));

  AbstractBasePtrList subargs;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    AbstractListPtr l_ptr = dyn_cast<AbstractList>(args_spec_list[i]);
    if (l_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Argument[" << i << "] of list_map should be a list.";
    }
    subargs.push_back(AbstractJoin(l_ptr->elements()));
  }
  AbstractBasePtr engin_exc = engine->Execute(fn, subargs);
  AbstractBasePtrList result;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    result.push_back(engin_exc);
  }
  return std::make_shared<AbstractList>(result);
}

AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: a fn, a list and an object of a subclass of a AbstractBase.
  MS_EXCEPTION_IF_NULL(engine);
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(op_name, args_spec_list, 0);
  AbstractListPtr lst = CheckArg<AbstractList>(op_name, args_spec_list, 1);
  AbstractBasePtr dflt = args_spec_list[2];

  AbstractBasePtr list_type = AbstractJoin(lst->elements());
  auto result1 = engine->Execute(fn, lst->elements());
  auto result2 = engine->Execute(fn, {dflt, list_type});
  MS_EXCEPTION_IF_NULL(result1);
  return result1->Join(result2);
}

AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  auto tuple_elements = input->elements();
  AbstractBasePtrList elem_list;
  (void)std::transform(tuple_elements.rbegin(), tuple_elements.rend(), std::back_inserter(elem_list),
                       [](const AbstractBasePtr &elem) { return elem->Clone(); });
  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr DoInferReduceShape(const AbstractTuplePtr &x_shape, const ValuePtr &x_shp_value,
                                   const ValueTuplePtr &axis_value_ptr, const PrimitivePtr &primitive) {
  size_t x_rank = x_shape->size();
  std::set<int> axis_set;
  auto axis_data = axis_value_ptr->value();
  if (axis_data.empty()) {
    int size = 1;
    AbstractBasePtrList values(x_rank, std::make_shared<AbstractScalar>(size));
    return std::make_shared<AbstractTuple>(values);
  }

  for (auto &elem : axis_data) {
    int e_value = CheckAxis(primitive->name(), elem, -SizeToInt(x_rank), SizeToInt(x_rank) - 1);
    (void)axis_set.insert(e_value);
  }

  auto x_shp_data = x_shp_value->cast<ValueTuplePtr>()->value();
  if (x_shp_data.size() < x_rank) {
    MS_LOG(EXCEPTION) << "x_shape_data.size() " << x_shp_data.size() << " less than x_shape.size() " << x_rank;
  }
  AbstractBasePtrList values;
  for (size_t i = 0; i < x_rank; i++) {
    if (axis_set.count(SizeToInt(i)) || axis_set.count(SizeToInt(i) - SizeToInt(x_rank))) {
      auto axis_v = MakeValue(1);
      values.push_back(std::make_shared<AbstractScalar>(axis_v, axis_v->type()));
    } else {
      int dim_value = x_shp_data[i]->cast<Int32ImmPtr>()->value();
      auto dim = MakeValue(dim_value);
      values.push_back(std::make_shared<AbstractScalar>(dim, dim->type()));
    }
  }

  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: x_shape, axis
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);

  auto x_shp_value = shape_x->BuildValue();
  if (x_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << op_name
                      << " evaluator shape's data field can't be anything: " << args_spec_list[1]->ToString();
  }

  // Axis can be scalar, tuple or None
  AbstractTuplePtr axis = nullptr;
  if (args_spec_list[1]->isa<AbstractScalar>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is scalar";
    AbstractBasePtrList axis_list = {dyn_cast<AbstractScalar>(args_spec_list[1])};
    axis = std::make_shared<AbstractTuple>(axis_list);
  } else if (args_spec_list[1]->isa<AbstractTuple>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is tuple";
    axis = args_spec_list[1]->cast<AbstractTuplePtr>();
  } else {
    MS_LOG(EXCEPTION) << op_name << " evaluator second parameter should be a scalar or tuple, but got "
                      << args_spec_list[1]->ToString();
  }

  auto axis_value = axis->BuildValue();
  if (axis_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << op_name
                      << " evaluator shape's data field can't be anything: " << args_spec_list[1]->ToString();
  }
  auto axis_value_ptr = axis_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(axis_value_ptr);

  return DoInferReduceShape(shape_x, x_shp_value, axis_value_ptr, primitive);
}

AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTuplePtr div_shp = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  MS_LOG(INFO) << "DivShape input:" << shape_x->ToString() << ", div:" << div_shp->ToString();

  auto div_shp_value = div_shp->BuildValue();
  if (div_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << args_spec_list[0]->ToString();
  }

  auto shpx_value = shape_x->BuildValue();
  if (shpx_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << args_spec_list[1]->ToString();
  }

  if (div_shp->size() != shape_x->size()) {
    MS_LOG(EXCEPTION) << "tileshape elems shape must the same div_shp: " << div_shp->size()
                      << ", shapex: " << shape_x->size() << ".";
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();
  auto div_shp_data = div_shp_value->cast<ValueTuplePtr>()->value();
  AbstractBasePtrList values;

  for (size_t i = 0; i < div_shp_data.size(); i++) {
    if (div_shp_data[i]->cast<Int32ImmPtr>() == nullptr) {
      MS_LOG(EXCEPTION) << "div_shp_shape data should be an int32 number, but it's " << args_spec_list[1]->ToString();
    }
    int shapex_value = GetValue<int>(shpx_data[i]);
    int div_value = GetValue<int>(div_shp_data[i]);
    MS_LOG(DEBUG) << "div_shp_shape data shapex_value :" << shapex_value << " div_value: " << div_value;
    if (div_value == 0) {
      MS_LOG(EXCEPTION) << "error: division value should not be 0!";
    }
    if ((shapex_value % div_value) != 0) {
      MS_LOG(EXCEPTION) << "div_shp_shape data shapex must div int:" << shapex_value << " div_value: " << div_value;
    }

    int result = shapex_value / div_value;
    auto result_v = MakeValue(result);
    values.push_back(std::make_shared<AbstractScalar>(result_v, result_v->type()));
  }

  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplTuple2Array(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  py::tuple data_tuple = ValuePtrToPyData(input->BuildValue());
  py::array data = py::array(data_tuple);
  auto tensor = std::make_shared<tensor::Tensor>(data);
  auto ret = tensor->ToAbstract();
  ret->set_value(tensor);
  MS_LOG(DEBUG) << "Tuple2arry result AbstractTensor: " << ret->ToString();
  return ret;
}

AbstractBasePtr InferImplShapeMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  // example: tuple = (1, 2, 3), shape_mul(tuple) = 1*2*3 = 6
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  auto shpx_value = shape_x->BuildValue();
  if (shpx_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << shape_x->ToString();
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();

  int result = 1;
  for (size_t i = 0; i < shpx_data.size(); i++) {
    int value = GetValue<int>(shpx_data[i]);
    IntMulWithOverflowCheck(result, value, &result);
  }

  auto result_v = MakeValue(result);
  MS_LOG(DEBUG) << "shape mul result:" << result_v->ToString();
  return std::make_shared<AbstractScalar>(result_v, result_v->type());
}

template <typename T>
AbstractBasePtr InferImplTupleOrListEqual(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples or two lists.
  CheckArgsSize(op_name, args_spec_list, 2);
  auto input_x = CheckArg<T>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<T>(op_name, args_spec_list, 1);

  ValuePtr x_value = input_x->BuildValue();
  ValuePtr y_value = input_y->BuildValue();
  return std::make_shared<AbstractScalar>(*x_value == *y_value);
}

AbstractBasePtr InferImplTupleEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractList>(primitive->name(), args_spec_list);
}

struct SlideInfo {
  int start;
  int step;
  int stop;
};

void CalcSlidePara(const AbstractBasePtrList &args_spec_list, SlideInfo *slide) {
  int arg1 = 0;
  int arg2 = 0;
  if (!args_spec_list.empty()) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    auto arg_value = args_spec_list[0]->BuildValue();
    if (!arg_value->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int32 number.";
    }
    arg1 = GetValue<int>(arg_value);
  }

  if (args_spec_list.size() >= 2) {
    MS_EXCEPTION_IF_NULL(args_spec_list[1]);
    auto arg_value = args_spec_list[1]->BuildValue();
    if (!arg_value->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int32 number.";
    }
    arg2 = GetValue<int>(arg_value);
  }

  if (args_spec_list.size() == 3) {
    MS_EXCEPTION_IF_NULL(args_spec_list[2]);
    auto arg_value = args_spec_list[2]->BuildValue();
    if (!arg_value->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int32 number.";
    }
    slide->step = GetValue<int>(arg_value);
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 2) {
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 1) {
    slide->stop = arg1;
  }
}

AbstractBasePtr InferImplMakeRange(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "Cannot make range from empty input.";
  }

  if (args_spec_list.size() > 3) {
    MS_LOG(EXCEPTION) << "Error args size of make range operational.";
  }

  SlideInfo slide = {0, 1, 0};
  CalcSlidePara(args_spec_list, &slide);

  if (slide.step == 0) {
    MS_LOG(EXCEPTION) << "Error, step value is 0.";
  }

  AbstractBasePtrList args;
  if (slide.start <= slide.stop) {
    if (slide.step <= 0) {
      MS_LOG(EXCEPTION) << "Error slice[" << slide.start << ", " << slide.stop << ", " << slide.step << "]";
    }
    for (int i = slide.start; i < slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
    }
  } else {
    if (slide.step >= 0) {
      MS_LOG(EXCEPTION) << "Error slice[" << slide.start << ", " << slide.stop << ", " << slide.step << "]";
    }
    for (int i = slide.start; i > slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
    }
  }

  return std::make_shared<AbstractTuple>(args);
}

AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Clone();
}
}  // namespace abstract
}  // namespace mindspore
