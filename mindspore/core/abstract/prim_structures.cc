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

#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"

namespace mindspore {
namespace abstract {
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
    auto key_string = GetValue<std::string>(keyPtr);
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
  auto key_string = GetValue<std::string>(keyPtr);
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
  auto key_input = GetValue<std::string>(key_value);
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
  AbstractBasePtrList slice_args;
  for (size_t index = 0; index < args_size; index++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[index]);
    if (args_spec_list[index]->isa<AbstractNone>()) {
      slice_args.push_back(args_spec_list[index]);
    } else if (args_spec_list[index]->isa<AbstractScalar>()) {
      ValuePtr scalar_value = args_spec_list[index]->cast<AbstractScalarPtr>()->BuildValue();
      if (scalar_value->isa<IntergerImm>()) {
        slice_args.push_back(args_spec_list[index]);
      } else if (scalar_value->isa<BoolImm>()) {
        ValuePtr scalar_index = MakeValue(static_cast<int64_t>(scalar_value->cast<BoolImmPtr>()->value()));
        slice_args.push_back(scalar_index->ToAbstract());
      } else {
        MS_EXCEPTION(TypeError) << "MakeSlice eval " << index
                                << " the input scalar type should be int or bool, but got " << scalar_value->ToString();
      }
    } else if (args_spec_list[index]->isa<AbstractTensor>()) {
      auto arg = args_spec_list[index]->cast<AbstractTensorPtr>();
      TypePtr tensor_dtype = arg->element()->BuildType();

      auto value = arg->BuildValue()->cast<tensor::TensorPtr>();
      if (value == nullptr) {
        MS_EXCEPTION(TypeError) << "MakeSlice eval the input tensor must be a const tensor.";
      }
      if (value->DataSize() != 1) {
        MS_EXCEPTION(TypeError) << "MakeSlice eval the input tensor must contain only one element, but got "
                                << value->ToString() << " has " << value->DataSize() << " elements.";
      }

      if (tensor_dtype->isa<Bool>()) {
        auto *bool_value = static_cast<bool *>(value->data_c());
        slice_args.push_back(MakeValue((static_cast<int64_t>(*bool_value)))->ToAbstract());
      } else if (tensor_dtype->isa<Int>()) {
        auto *int_value = static_cast<int64_t *>(value->data_c());
        slice_args.push_back(MakeValue((*int_value))->ToAbstract());
      } else {
        MS_EXCEPTION(TypeError) << "MakeSlice eval the input tensor type must be int or bool, but got "
                                << tensor_dtype->ToString();
      }
    } else {
      MS_EXCEPTION(TypeError) << "MakeSlice eval " << index << " inputs should scalar, None or Tensor, but got"
                              << args_spec_list[index]->ToString();
    }
  }
  // Slice: start, end, step
  return std::make_shared<AbstractSlice>(slice_args[0], slice_args[1], slice_args[2]);
}

template <typename T>
AbstractBasePtr InferTupleOrListGetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list and a scalar whose value is an int32 number.
  CheckArgsSize(op_name, args_spec_list, 2);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  if (!index_value->isa<Int64Imm>()) {
    // when index_value is an AnyValue and args_spec_list[0] is a scalar, try to return the type of the first element
    //  and continue
    if (dyn_cast<AbstractScalar>(queue->elements()[0]) != nullptr) {
      return std::make_shared<AbstractScalar>(queue->elements()[0]->BuildType());
    }
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be an int64 number, but got "
                             << index_value->ToString();
  }
  auto idx_v = GetValue<int64_t>(index_value);
  std::size_t nelems = queue->elements().size();
  if (idx_v >= SizeToLong(nelems) || idx_v < -SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be in range[-" << SizeToLong(nelems) << ", "
                             << SizeToLong(nelems) << "), but got " << idx_v << ".";
  }

  std::size_t uidx_v = 0;
  if (idx_v >= 0) {
    uidx_v = LongToSize(idx_v);
  } else {
    uidx_v = LongToSize(idx_v + SizeToLong(nelems));
  }
  return queue->elements()[uidx_v];
}

template <typename T>
AbstractBasePtr InferTupleOrListSetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list, a scalar whose value is an int64 number and an object of a subclass of AbstractBase.
  CheckArgsSize(op_name, args_spec_list, 3);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  if (!index_value->isa<Int64Imm>()) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be an int64 number, but got "
                             << index_value->ToString();
  }
  auto idx_v = GetValue<int64_t>(index_value);
  AbstractBasePtrList elements = queue->elements();
  std::size_t nelems = elements.size();
  int64_t idx_t = idx_v >= 0 ? idx_v : idx_v + SizeToLong(nelems);
  if (idx_t < 0 || idx_t >= SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator the index: " << idx_v << " to set out of range: [-" << nelems
                             << "," << nelems - 1 << "].";
  }
  size_t uidx_v = LongToSize(idx_t);
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
    MS_EXCEPTION(KeyError) << "The key " << key_str << " does not exist in the dict:" << args_spec_list[0]->ToString();
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
  auto key_str = GetValue<std::string>(key_value);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [key_str](const AbstractAttribute &item) { return item.first == key_str; });

  MS_EXCEPTION_IF_NULL(args_spec_list[2]);
  auto new_ele = std::make_pair(key_str, args_spec_list[2]);
  if (it != dict_elems.end()) {
    int64_t index = it - dict_elems.begin();
    dict_elems[LongToSize(index)] = new_ele;
  } else {
    dict_elems.push_back(new_ele);
  }
  return std::make_shared<AbstractDictionary>(dict_elems);
}

AbstractBasePtr InferImplDictGetKeys(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  AbstractBasePtrList keys;
  std::transform(dict_elems.begin(), dict_elems.end(), std::back_inserter(keys),
                 [](const AbstractAttribute &item) { return std::make_shared<AbstractScalar>(item.first); });
  return std::make_shared<AbstractTuple>(keys);
}

AbstractBasePtr InferImplDictGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  AbstractBasePtrList values;
  std::transform(dict_elems.begin(), dict_elems.end(), std::back_inserter(values),
                 [](const AbstractAttribute &item) { return item.second; });
  return std::make_shared<AbstractTuple>(values);
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

AbstractBasePtr InferImplTupleLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplArrayLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg_abs = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto shape = arg_abs->BuildShape()->cast<ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->shape().empty()) {
    MS_EXCEPTION(TypeError) << "Not support len of a 0-D tensor.";
  }
  return std::make_shared<AbstractScalar>(shape->shape()[0]);
}
}  // namespace abstract
}  // namespace mindspore
