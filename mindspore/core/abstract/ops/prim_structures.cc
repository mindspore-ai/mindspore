/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "abstract/ops/infer_functions.h"
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
  constexpr int args_spec_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
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
    (void)key_value.emplace_back(key_string, value_list[index]);
  }
  return std::make_shared<AbstractDictionary>(key_value);
}

AbstractBasePtr InferImplMakeKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a string and an object of a subclass of AbstractBase.
  const std::string op_name = primitive->name();
  constexpr int args_spec_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);

  ValuePtr keyPtr = key->BuildValue();
  MS_EXCEPTION_IF_NULL(keyPtr);
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
  constexpr int args_spec_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractKeywordArgPtr kwarg = CheckArg<AbstractKeywordArg>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
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

template <typename T>
AbstractBasePtr InferTupleOrListGetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list and a scalar whose value is an int32 number.
  constexpr int args_spec_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  MS_EXCEPTION_IF_NULL(index_value);
  if (!index_value->isa<Int64Imm>()) {
    // when index_value is an AnyValue and args_spec_list[0] is a scalar, try to return the type of the first element
    //  and continue
    if (dyn_cast<AbstractScalar>(queue->elements()[0]) != nullptr) {
      return std::make_shared<AbstractScalar>(queue->elements()[0]->BuildType());
    }
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be an int64 number, but got " << index->ToString();
  }
  auto index_int64_value = GetValue<int64_t>(index_value);
  std::size_t nelems = queue->elements().size();
  if (index_int64_value >= SizeToLong(nelems) || index_int64_value < -SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be in range[-" << SizeToLong(nelems) << ", "
                             << SizeToLong(nelems) << "), but got " << index_int64_value << ".";
  }

  std::size_t index_unsigned_value = 0;
  if (index_int64_value >= 0) {
    index_unsigned_value = LongToSize(index_int64_value);
  } else {
    index_unsigned_value = LongToSize(index_int64_value + SizeToLong(nelems));
  }
  MS_LOG(DEBUG) << "GetItem use flags, index: " << index_unsigned_value << ", for " << queue->ToString();
  SetSequenceElementsUseFlags(queue, index_unsigned_value, true);
  return queue->elements()[index_unsigned_value];
}

template <typename T>
AbstractBasePtr InferTupleOrListSetItem(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list, a scalar whose value is an int64 number and an object of a subclass of AbstractBase.
  constexpr int args_spec_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  auto queue = CheckArg<T>(op_name, args_spec_list, 0);
  AbstractScalarPtr index = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr index_value = index->BuildValue();
  MS_EXCEPTION_IF_NULL(index_value);
  if (!index_value->isa<Int64Imm>()) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be an int64 number, but got "
                             << index_value->ToString();
  }
  auto index_int64_value = GetValue<int64_t>(index_value);
  AbstractBasePtrList elements = queue->elements();
  std::size_t nelems = elements.size();
  int64_t index_positive_value = index_int64_value >= 0 ? index_int64_value : index_int64_value + SizeToLong(nelems);
  if (index_positive_value < 0 || index_positive_value >= SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator the index: " << index_int64_value << " to set out of range: [-"
                             << nelems << "," << (nelems - 1) << "].";
  }
  size_t index_unsigned_value = LongToSize(index_positive_value);
  constexpr int target_value_index = 2;
  elements[index_unsigned_value] = args_spec_list[target_value_index];
  MS_LOG(DEBUG) << "SetItem use flags, index: " << index_unsigned_value << ", for " << queue->ToString();
  return std::make_shared<T>(elements, queue->sequence_nodes());
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
  const std::string op_name = primitive->name();
  // dict[key] mean the size of args_spec_list is 2.
  // dict.get('key', default_value=None) mean the size of args_spec_list is 2 too, the key will check in dict_get.
  constexpr int subscript_args_size = 2;
  if (args_spec_list.size() != subscript_args_size) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the number of input should be " << subscript_args_size
                      << ", but got " << args_spec_list.size();
  }
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << key_value->ToString();
  }
  auto key_str = GetValue<std::string>(key_value);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [key_str](const AbstractAttribute &item) { return item.first == key_str; });
  if (it == dict_elems.end()) {
    // For dict[key], if key is not exist, will raise a KeyError exception.
    // For dict.get('key', default=None), if key is not exist, will return the default value during dict_get.
    MS_EXCEPTION(KeyError) << "The key " << key_value->ToString()
                           << " does not exist in the dict:" << args_spec_list[0]->BuildValue()->ToString();
  }
  return it->second;
}

AbstractBasePtr InferImplDictSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict and a scalar whose value is a string and an object of a subclass of AbstractBase.
  const std::string op_name = primitive->name();
  constexpr int args_spec_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  AbstractScalarPtr key = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
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
  constexpr int args_spec_size = 1;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
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
  constexpr int args_spec_size = 1;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  AbstractBasePtrList values;
  std::transform(dict_elems.begin(), dict_elems.end(), std::back_inserter(values),
                 [](const AbstractAttribute &item) { return item.second; });
  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplDictItems(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a dict.
  const std::string op_name = primitive->name();
  constexpr int args_spec_size = 1;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
  AbstractDictionaryPtr dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 0);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  AbstractBasePtrList items;
  (void)std::transform(dict_elems.begin(), dict_elems.end(), std::back_inserter(items),
                       [](const AbstractAttribute &item) {
                         return std::make_shared<AbstractTuple>(
                           AbstractBasePtrList{std::make_shared<AbstractScalar>(item.first), item.second});
                       });
  return std::make_shared<AbstractList>(items);
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
  constexpr int args_spec_size = 1;
  CheckArgsSize(op_name, args_spec_list, args_spec_size);
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
