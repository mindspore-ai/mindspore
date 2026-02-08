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

#include <string>
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "abstract/param_validator.h"
#include "infer/deprecated_infer/infer_functions.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/compile_config.h"

namespace {
constexpr auto kRankSize = "rank_size";
constexpr auto kDependInputAbs = "depend_input_abs";
}  // namespace

namespace mindspore {

namespace abstract {
AbstractBasePtr InferImplidentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_abs_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  return args_abs_list[0];
}

AbstractBasePtr InferImplEnvironAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_abs_list) {
  // args: Three objects of a subclass of AbstractBase, env, key, dflt(default).
  constexpr auto environ_add_input_size = 2;
  CheckArgsSize(primitive->name(), args_abs_list, environ_add_input_size);
  return std::make_shared<AbstractScalar>(kValueAny, std::make_shared<EnvType>());
}

AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  // args: Two objects of a subclass of AbstractBase, key and value.
  constexpr auto state_setitem_input_size = 2;
  CheckArgsSize(primitive->name(), args_abs_list, state_setitem_input_size);

  TypePtr type = args_abs_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kObjectTypeRefKey && type->type_id() != kObjectTypeSymbolicKeyType) {
    MS_LOG(EXCEPTION) << "First input of StateSetItem should be a RefKey or SymbolicKeyType but a " << type->ToString();
  }
  return std::make_shared<AbstractScalar>(kValueAny, kBool);
}

namespace {
void UpdateDependInplaceAbstract(const AbstractBasePtr &src, const AbstractBasePtr &dst) {
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(dst);

  dst->set_inplace_abstract(src->inplace_abstract());

  auto src_seq = src->cast_ptr<AbstractSequence>();
  if (src_seq != nullptr) {
    auto dst_seq = dst->cast_ptr<AbstractSequence>();
    MS_EXCEPTION_IF_NULL(dst_seq);
    if (src_seq->elements().size() != dst_seq->elements().size()) {
      MS_LOG(INFO) << "The size of src and dst sequence are different, skip updating inplace abstract, src size: "
                   << src_seq->elements().size() << ", dst size: " << dst_seq->elements().size();
      MS_LOG(DEBUG) << "src sequence: " << src_seq->ToString() << ",\n dst sequence: " << dst_seq->ToString();
      return;
    }
    for (size_t i = 0; i < src_seq->elements().size(); ++i) {
      UpdateDependInplaceAbstract(src_seq->elements()[i], dst_seq->elements()[i]);
    }
  }
}
}  // namespace

AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_abs_list) {
  constexpr auto depend_input_size = 2;
  CheckArgsSize(primitive->name(), args_abs_list, depend_input_size);

  // If the dependent has a value, just return depended node.
  // If depended node is not Any, the dependent maybe eliminated.
  auto dependant_abstract = args_abs_list[1];
  auto dependant_value = dependant_abstract->BuildValue();
  MS_EXCEPTION_IF_NULL(dependant_value);
  if (!dependant_value->ContainsValueAny()) {
    return args_abs_list[0];
  }
  auto depends = args_abs_list[0];

  if (depends->isa<AbstractRefTensor>()) {
    auto abs_ref = depends->cast<AbstractRefPtr>();
    auto tensor_abs = abs_ref->ref();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    auto res =
      std::make_shared<AbstractRefTensor>(tensor_abs->Broaden()->cast<AbstractTensorPtr>(), abs_ref->ref_key_value());
    res->cast<AbstractRefPtr>()->set_ref_tensor_type(abs_ref->ref_tensor_type());
    return res;
  }

  // Note: Broaden() may delete inplace_abstract.
  auto depends_abs = depends->Broaden();  // Avoid eliminating the dependent node.
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) &&
      common::GetCompileConfig("GRAD_FOR_SCALAR") != "1") {
    // For scalar, need to set value to kValueAny, because broaden scalar will not change the value.
    if (depends_abs->isa<AbstractScalar>()) {
      depends_abs->set_value(kValueAny);
    }
  }
  UpdateDependInplaceAbstract(depends, depends_abs);
  depends_abs->set_user_data<abstract::AbstractBase>(kDependInputAbs, depends);
  return depends_abs;
}

AbstractBasePtr InferImplUpdateState(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION) << primitive->name() << " input args size should be at least 1, but got 0";
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  return args_abs_list[0]->Broaden();
}

AbstractBasePtr InferImplVirtualViewGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tensors, VirtualViewGrad(view_input, view_output)
  constexpr size_t size_expected = 2;
  if (args_abs_list.size() < size_expected) {
    MS_LOG(EXCEPTION) << primitive->name() << " input args size should be at least 2, but got " << args_abs_list.size();
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  return args_abs_list[0];
}

AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  constexpr size_t size_expected = 3;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  auto indices = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto values = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  auto dense_shape = CheckArg<AbstractTuple>(op_name, args_abs_list, 2);

  auto indices_dtype = indices->element()->BuildType();
  if (!indices_dtype->isa<Int>()) {
    MS_EXCEPTION(TypeError) << "The dtype of indices must be a Int, but got " << indices_dtype->ToString();
  }
  auto indices_shp = indices->shape()->shape();
  auto values_shp = values->shape()->shape();
  auto is_values_dynamic = IsDynamic(values_shp);
  if (!IsDynamic(indices_shp) && !is_values_dynamic) {
    if (indices_shp.size() != 1) {
      MS_EXCEPTION(TypeError) << "Indices must be a 1 dimension tensor, but got a " << indices_shp.size()
                              << " dimension tensor";
    }
    if (indices_shp[0] != values_shp[0]) {
      MS_EXCEPTION(TypeError) << "The first dimension of indices must be the same with the first dimension of values "
                              << values_shp[0] << ", but got " << indices_shp[0];
    }
  }

  for (const auto &elem_type : dense_shape->ElementsType()) {
    if (!elem_type->isa<Int>()) {
      MS_EXCEPTION(TypeError) << "The element type of dense_shape must be Int, but got " << elem_type->ToString();
    }
  }
  auto dense_shape_value = dense_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(dense_shape_value);
  auto dense_shape_valuetuple = dense_shape_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(dense_shape_valuetuple);
  auto shp = dense_shape_valuetuple->value();
  ShapeVector dense_shape_vec;
  (void)std::transform(std::begin(shp), std::end(shp), std::back_inserter(dense_shape_vec),
                       [](const ValuePtr &e) -> int64_t {
                         auto elem = GetValue<int64_t>(e);
                         return elem;
                       });
  if (dense_shape_vec.size() != values_shp.size() && !is_values_dynamic) {
    MS_EXCEPTION(TypeError) << "The size of dense_shape must be the same with the dimension of values "
                            << values_shp.size() << ", but got " << dense_shape_valuetuple->size();
  }
  for (size_t i = 0; i < dense_shape_vec.size(); i++) {
    if (dense_shape_vec[i] < 0) {
      MS_EXCEPTION(TypeError) << "The " << i << "th element of dense_shape must be positive, but got "
                              << dense_shape_vec[i];
    }
    // The 0th mode might be less or exceed dense_shape[0] due to duplicated selection
    if (!is_values_dynamic && i != 0 && dense_shape_vec[i] != values_shp[i]) {
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
                                            const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->values());
  return row_tensor->values();
}

AbstractBasePtr InferImplRowTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->indices());
  return row_tensor->indices();
}

AbstractBasePtr InferImplRowTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(row_tensor->dense_shape());
  return row_tensor->dense_shape();
}

AbstractBasePtr InferImplRowTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  // Inputs: row tensor and tensor.
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 2;
  CheckArgsSize(op_name, args_abs_list, args_size);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_abs_list, 0);
  auto tensor = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  MS_EXCEPTION_IF_NULL(row_tensor->dense_shape());
  MS_EXCEPTION_IF_NULL(tensor->shape());
  return args_abs_list[0];
}

AbstractBasePtr InferImplAllReduce(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(x->shape()->shape()));
}

AbstractBasePtr InferImplReduceScatter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto tmp_shape = x->shape()->shape();
  if (!primitive->HasAttr(kRankSize)) {
    MS_LOG(EXCEPTION) << "Primitive don't have rank_size attr";
  }
  auto rank_size = GetValue<int64_t>(primitive->GetAttr(kRankSize));
  if (tmp_shape.empty()) {
    MS_LOG(EXCEPTION) << "shape size is 0";
  }
  tmp_shape[0] = LongMulWithOverflowCheck(tmp_shape[0], rank_size);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(tmp_shape));
}

AbstractBasePtr InferImplIsDimUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, input_size);
  auto abs = args_abs_list[0];
  if (abs->isa<AbstractAny>()) {
    return std::make_shared<AbstractAny>();
  }
  if (!abs->isa<AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be tuple but got " << abs->ToString();
  }
  auto abs_seq = abs->cast<AbstractSequencePtr>();
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(abs_seq->dynamic_len()), kBool);
}

AbstractBasePtr InferImplIsTensorBoolCond(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, input_size);
  auto abs = args_abs_list[0];
  if (!abs->isa<AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be a tensor but got " << abs->ToString();
  }

  auto tensor_ptr = abs->cast<AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto build_shape = tensor_ptr->GetShape();
  MS_EXCEPTION_IF_NULL(build_shape);
  if (build_shape->IsDimUnknown()) {
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(true), kBool);
  }
  auto shape_ptr = build_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto shape = shape_ptr->shape();
  if (shape.size() == 0) {
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(true), kBool);
  }
  if (shape.size() == 1 && (shape[0] == abstract::Shape::kShapeDimAny || shape[0] == 1)) {
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(true), kBool);
  }
  MS_EXCEPTION(ValueError) << "Only tensor which shape is () or (1,) can be converted to bool, "
                           << "but got tensor shape is " << build_shape->ToString();
}

AbstractBasePtr InferImplIsShapeUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, input_size);
  auto abs = args_abs_list[0];
  if (!abs->isa<AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be tuple or list but got " << abs->ToString();
  }
  auto abs_seq = abs->cast<AbstractSequencePtr>();
  bool is_shape_unknown = false;
  if (abs_seq->dynamic_len()) {
    is_shape_unknown = true;
  } else {
    auto &elements = abs_seq->elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      auto cur = elements[i];
      MS_EXCEPTION_IF_NULL(cur);
      auto cur_val = cur->BuildValue();
      MS_EXCEPTION_IF_NULL(cur_val);
      if (cur_val->ContainsValueAny()) {
        is_shape_unknown = true;
        break;
      }
    }
  }
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(is_shape_unknown), kBool);
}

AbstractBasePtr InferImplIsElementUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, input_size);
  auto abs = args_abs_list[0];
  if (!abs->isa<AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be tuple or list but got " << abs->ToString();
  }
  auto abs_seq = abs->cast<AbstractSequencePtr>();
  if (!abs_seq->dynamic_len()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be variable length sequence.";
  }
  bool is_element_unknown = (abs_seq->dynamic_len_element_abs() == nullptr);
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(is_element_unknown), kBool);
}

AbstractBasePtr InferImplLoad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_abs_list) {
  // Inputs: Ref/Tensor, universal
  constexpr auto load_input_size = 2;
  CheckArgsSize(primitive->name(), args_abs_list, load_input_size);
  auto ref_abs = dyn_cast<abstract::AbstractRefTensor>(args_abs_list[0]);
  if (ref_abs != nullptr) {
    // Return tensor value if input is Ref.
    return ref_abs->CloneAsTensor();
  }
  return args_abs_list[0]->Broaden();
}

AbstractBasePtr InferImplTransData(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  auto output = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(output);
  return output;
}

AbstractBasePtr InferImplTensorMove(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_abs_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  auto output = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(output);
  return output;
}

// Infer for MapTensor.default_value.
AbstractBasePtr InferImplMapTensorGetDefaultValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const AbstractBasePtrList &args_abs_list) {
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  const auto &arg = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->default_value());
}
// Infer for MapTensor.permit_filter_value.
AbstractBasePtr InferImplMapTensorGetPermitFilterValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                       const AbstractBasePtrList &args_abs_list) {
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  const auto &arg = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->permit_filter_value());
}
// Infer for MapTensor.evict_filter_value.
AbstractBasePtr InferImplMapTensorGetEvictFilterValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                      const AbstractBasePtrList &args_abs_list) {
  CheckArgsSize(primitive->name(), args_abs_list, 1);
  const auto &arg = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->evict_filter_value());
}
}  // namespace abstract
}  // namespace mindspore
