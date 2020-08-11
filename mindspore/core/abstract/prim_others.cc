/**
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

#include <string>
#include <sstream>

#include "ir/dtype.h"
#include "utils/ms_utils.h"
#include "base/core_ops.h"
#include "abstract/param_validator.h"
#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplIdentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0];
}

AbstractBasePtr InferImplEnvGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  // args: Three objects of a subclass of AbstractBase, env, key, dflt(default).
  CheckArgsSize(primitive->name(), args_spec_list, 3);
  auto key = args_spec_list[1];
  auto dflt = args_spec_list[2];
  TypePtr type = key->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kObjectTypeSymbolicKeyType) {
    MS_LOG(EXCEPTION) << "EnvGetItem evaluator args[1] should be a SymbolicKeyInstance but: " << key->ToString();
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_sparse = context->enable_sparse();
  if (enable_sparse && dflt->isa<AbstractTensor>()) {
    auto dflt_tensor = dflt->cast<AbstractTensorPtr>();
    return std::make_shared<AbstractUndetermined>(dflt_tensor->element()->Clone(), dflt_tensor->shape()->Clone());
  }

  if (!key->GetValueTrack()->isa<SymbolicKeyInstance>()) {
    return dflt;
  }
  ValuePtr key_value_ptr = key->GetValueTrack();
  MS_EXCEPTION_IF_NULL(key_value_ptr);
  auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
  auto expected = key_value_track->abstract();
  MS_EXCEPTION_IF_NULL(expected);
  (void)expected->Join(dflt);
  return expected;
}

AbstractBasePtr InferImplEnvSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // args: Three objects of a subclass of AbstractBase, env, key, dflt(default).
  CheckArgsSize(primitive->name(), args_spec_list, 3);

  auto key = args_spec_list[1];
  ValuePtr key_value_ptr = key->GetValueTrack();
  MS_EXCEPTION_IF_NULL(key_value_ptr);
  auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
  if (key_value_track == nullptr) {
    MS_LOG(EXCEPTION) << "EnvGetItem evaluator args[1] expected should be able to cast to SymbolicKeyInstancePtrbut: "
                      << key_value_ptr->ToString();
  }
  auto expected = key_value_track->abstract();
  MS_EXCEPTION_IF_NULL(expected);
  return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
}

AbstractBasePtr InferImplEnvAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // args: Three objects of a subclass of AbstractBase, env, key, dflt(default).
  CheckArgsSize(primitive->name(), args_spec_list, 2);
  return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
}

AbstractBasePtr InferImplMakeRefKey(const AnalysisEnginePtr &, const PrimitivePtr &prim, const AbstractBasePtrList &) {
  ValuePtr name_value = prim->GetAttr("tag");
  auto name = name_value->cast<StringImmPtr>();
  if (name == nullptr) {
    MS_LOG(EXCEPTION) << "MakeRefKey attr tag sould be a String " << name_value->ToString() << ".";
  }
  auto refkey = std::make_shared<RefKey>(name->value());
  if (refkey == nullptr) {
    MS_LOG(EXCEPTION) << "MakeRefKey std::make_shared<RefKey> failed";
  }
  return refkey->ToAbstract();
}

AbstractBasePtr InferImplMakeRef(const AnalysisEnginePtr &, const PrimitivePtr &,
                                 const AbstractBasePtrList &args_spec_list) {
  // arguments: key, value, target type(None if no target type)
  if (args_spec_list.size() != 3) {
    MS_LOG(EXCEPTION) << "make_ref evaluator requires 3 parameters, while the input size is " << args_spec_list.size()
                      << ".";
  }
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  ValuePtr tensor_target_v = args_spec_list[2]->BuildValue();
  if (type->type_id() != kObjectTypeRefKey) {
    MS_LOG(EXCEPTION) << "First input of make_ref should be a RefKey but a " << type->ToString();
  }
  auto need_cast = !tensor_target_v->isa<None>();
  if (need_cast && !tensor_target_v->isa<Type>()) {
    MS_LOG(EXCEPTION) << "Third input of make_ref should be a Type but a " << tensor_target_v->ToString();
  }
  TypePtr cast_target = tensor_target_v->cast<TypePtr>();
  return std::make_shared<AbstractRef>(args_spec_list[0], args_spec_list[1], need_cast, cast_target);
}

AbstractBasePtr InferImplGetRefKey(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  // arguments: value
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "get_ref_key requires 1 parameters, while the input size is " << args_spec_list.size() << ".";
  }
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  if (type->type_id() != kObjectTypeRef) {
    MS_LOG(EXCEPTION) << "First input of get_ref_key should be a Ref but a " << type->ToString();
  }
  return args_spec_list[0]->cast<AbstractRefPtr>()->ref();
}

AbstractBasePtr InferImplGetRefValue(const AnalysisEnginePtr &, const PrimitivePtr &,
                                     const AbstractBasePtrList &args_spec_list) {
  // arguments: value
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "get_ref_value requires 1 parameters, while the input size is " << args_spec_list.size()
                      << ".";
  }
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  if (type->type_id() != kObjectTypeRef) {
    return args_spec_list[0];
  }
  return args_spec_list[0]->cast<AbstractRefPtr>()->ref();
}

AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // args: Two objects of a subclass of AbstractBase, key and value.
  CheckArgsSize(primitive->name(), args_spec_list, 2);

  TypePtr type = args_spec_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kObjectTypeRefKey && type->type_id() != kObjectTypeSymbolicKeyType) {
    MS_LOG(EXCEPTION) << "First input of StateSetItem should be a RefKey or SymbolicKeyType but a " << type->ToString();
  }
  return std::make_shared<AbstractScalar>(kAnyValue, kBool);
}

AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << primitive->name() << " input args size should be at lest 1, but got 0";
  }
  auto depends = args_spec_list[0]->Broaden();
  return depends;
}

AbstractBasePtr InferImplControlDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // args: Two objects of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_spec_list, 2);
  auto arg_src = args_spec_list[0];
  auto arg_dst = args_spec_list[1];
  // control depend can not setup tuple of ops to tuple of ops dependency relation
  if (arg_src->isa<AbstractTuple>() && arg_dst->isa<AbstractTuple>()) {
    auto src_size = arg_src->cast<AbstractTuplePtr>()->size();
    auto dst_size = arg_src->cast<AbstractTuplePtr>()->size();
    if (src_size > 1 && dst_size > 1) {
      MS_LOG(EXCEPTION) << "Control depend can not setup operator dependcy relationship from tuple from tuple";
    }
  }
  return std::make_shared<AbstractScalar>(kAnyValue, kBool);
}

AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto values = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dense_shape = CheckArg<AbstractTuple>(op_name, args_spec_list, 2);

  auto indices_dtype = indices->element()->BuildType();
  if (!indices_dtype->isa<Int>()) {
    MS_EXCEPTION(TypeError) << "The dtype of indices must be a Int, but got " << indices_dtype->ToString();
  }
  auto indices_shp = indices->shape()->shape();
  if (indices_shp.size() != 1) {
    MS_EXCEPTION(TypeError) << "Indices must be a 1 dimension tensor, but got a " << indices_shp.size()
                            << " dimension tensor";
  }
  auto values_shp = values->shape()->shape();
  if (indices_shp[0] != values_shp[0]) {
    MS_EXCEPTION(TypeError) << "The first dimension of indices must be the same with the first dimension of values "
                            << values_shp[0] << ", but got " << indices_shp[0];
  }

  for (auto elem_type : dense_shape->ElementsType()) {
    if (!elem_type->isa<Int>()) {
      MS_EXCEPTION(TypeError) << "The element type of dense_shape must be Int, but got " << elem_type->ToString();
    }
  }
  auto dense_shape_value = dense_shape->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(dense_shape_value);
  auto shp = dense_shape_value->value();
  std::vector<int> dense_shape_vec;
  (void)std::transform(std::begin(shp), std::end(shp), std::back_inserter(dense_shape_vec),
                       [](const ValuePtr &e) -> int {
                         auto elem = GetValue<int>(e);
                         return elem;
                       });
  if (dense_shape_vec.size() != values_shp.size()) {
    MS_EXCEPTION(TypeError) << "The size of dense_shape must be the same with the dimension of values "
                            << values_shp.size() << ", but got " << dense_shape_value->size();
  }
  for (size_t i = 0; i < dense_shape_vec.size(); i++) {
    if (dense_shape_vec[i] < 0) {
      MS_EXCEPTION(TypeError) << "The " << i << "th element of dense_shape must be positive, but got "
                              << dense_shape_vec[i];
    }
    // The 0th mode might be less or exceed dense_shape[0] due to duplicated selection
    if (i != 0 && dense_shape_vec[i] != values_shp[i]) {
      MS_EXCEPTION(TypeError) << "The " << i << "th element of dense_shape must be same with the " << i
                              << "th dimension of values " << values_shp[i] << ", but got " << dense_shape_vec[i];
    }
  }
  auto ret = std::make_shared<AbstractRowTensor>(values->element()->BuildType(), dense_shape_vec);
  ret->set_indices(indices);
  ret->set_values(values);
  ret->set_dense_shape(dense_shape);
  return ret;
}

AbstractBasePtr InferImplRowTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->values());
  return row_tensor->values();
}

AbstractBasePtr InferImplRowTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->indices());
  return row_tensor->indices();
}

AbstractBasePtr InferImplRowTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->dense_shape());
  return row_tensor->dense_shape();
}

AbstractBasePtr InferImplMakeSparseTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto values = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dense_shape = CheckArg<AbstractTuple>(op_name, args_spec_list, 2);

  auto indices_dtype = indices->element()->BuildType();
  if (!indices_dtype->isa<Int>()) {
    MS_EXCEPTION(TypeError) << "The dtype of indices must be a Int, but got " << indices_dtype->ToString();
  }
  auto indices_shp = indices->shape()->shape();
  if (indices_shp.size() != 2) {
    MS_EXCEPTION(TypeError) << "Indices must be a 2 dimension tensor, but got a " << indices_shp.size()
                            << " dimension tensor";
  }
  auto values_shp = values->shape()->shape();
  if (values_shp.size() != 1) {
    MS_EXCEPTION(TypeError) << "Values must be a 1 dimension tensor, but got a " << values_shp.size()
                            << " dimension tensor";
  }
  if (indices_shp[0] != values_shp[0]) {
    MS_EXCEPTION(TypeError) << "The first dimension of indices must be the same with the first dimension of values "
                            << values_shp[0] << ", but got " << indices_shp[0];
  }

  for (auto elem_type : dense_shape->ElementsType()) {
    if (!elem_type->isa<Int>()) {
      MS_EXCEPTION(TypeError) << "The element type of dense_shape must be Int, but got " << elem_type->ToString();
    }
  }
  auto dense_shape_value = dense_shape->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(dense_shape_value);
  auto shp = dense_shape_value->value();
  std::vector<int> dense_shape_vec;
  (void)std::transform(std::begin(shp), std::end(shp), std::back_inserter(dense_shape_vec),
                       [](const ValuePtr &e) -> int {
                         auto elem = GetValue<int>(e);
                         return elem;
                       });
  if (IntToSize(indices_shp[1]) != dense_shape_vec.size()) {
    MS_EXCEPTION(TypeError) << "The size of dense_shape must be equal with the second dimension of indices "
                            << indices_shp[1] << ", but got " << dense_shape_vec.size();
  }
  for (auto dense_shape_elem : dense_shape_vec) {
    if (dense_shape_elem < 0) {
      MS_EXCEPTION(TypeError) << "The element of dense_shape must be positive, but got "
                              << dense_shape_value->ToString();
    }
  }
  auto ret = std::make_shared<AbstractSparseTensor>(values->element()->BuildType(), dense_shape_vec);
  ret->set_indices(indices);
  ret->set_values(values);
  ret->set_dense_shape(dense_shape);
  return ret;
}

AbstractBasePtr InferImplSparseTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto sparse_tensor = CheckArg<AbstractSparseTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(sparse_tensor->values());
  return sparse_tensor->values();
}

AbstractBasePtr InferImplSparseTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto sparse_tensor = CheckArg<AbstractSparseTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(sparse_tensor->indices());
  return sparse_tensor->indices();
}

AbstractBasePtr InferImplSparseTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto sparse_tensor = CheckArg<AbstractSparseTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(sparse_tensor->dense_shape());
  return sparse_tensor->dense_shape();
}
}  // namespace abstract
}  // namespace mindspore
