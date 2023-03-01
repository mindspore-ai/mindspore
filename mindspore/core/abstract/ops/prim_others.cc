/**
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

#include <string>

#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "abstract/param_validator.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/utils.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "utils/shape_utils.h"
#include "ops/real_div.h"
#include "ops/add.h"
#include "ops/mul.h"
#include "ops/sub.h"
#include "ops/square.h"
#include "ops/assign.h"

namespace {
constexpr auto kRankSize = "rank_size";
}  // namespace

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplIdentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0];
}

AbstractBasePtr InferImplEnvironCreate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // args: None.
  CheckArgsSize(primitive->name(), args_spec_list, 0);
  static const AbstractBasePtr abs_env = std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  return abs_env;
}

AbstractBasePtr InferImplEnvironGet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  // args: Three objects of a subclass of AbstractBase, env, key, default_value(default).
  CheckArgsSize(primitive->name(), args_spec_list, kSizeThree);
  auto key = args_spec_list[kIndexOne];
  auto default_value = args_spec_list[kIndexTwo];
  TypePtr type = key->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kObjectTypeSymbolicKeyType) {
    MS_LOG(EXCEPTION) << "EnvironGet evaluator args[1] should be a SymbolicKeyInstance but: " << key->ToString();
  }

  MS_LOG(DEBUG) << "key: " << key->ToString() << ", value: " << default_value->ToString();
  if (default_value->isa<AbstractTensor>() && EnvSetSparseResultMgr::GetInstance().Get()) {
    auto tensor_value = default_value->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_value);
    return std::make_shared<AbstractUndetermined>(tensor_value->element()->Clone(), tensor_value->shape()->Clone());
  }

  if (!key->GetValueTrack()->isa<SymbolicKeyInstance>()) {
    return default_value;
  }
  ValuePtr key_value_ptr = key->GetValueTrack();
  MS_EXCEPTION_IF_NULL(key_value_ptr);
  auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
  auto expected = key_value_track->abstract();
  MS_EXCEPTION_IF_NULL(expected);
  (void)expected->Join(default_value);
  // If expected is AbstractRef, return it's AbstractTensor as Value type other than Reference type.
  if (expected->isa<AbstractRefTensor>()) {
    const auto &abs_ref = expected->cast<AbstractRefPtr>();
    MS_EXCEPTION_IF_NULL(abs_ref);
    return abs_ref->CloneAsTensor();
  }
  return expected;
}

AbstractBasePtr InferImplEnvironSet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // args: Three objects of a subclass of AbstractBase, env, key, value.
  CheckArgsSize(primitive->name(), args_spec_list, kSizeThree);

  auto key = args_spec_list[kIndexOne];
  ValuePtr key_value_ptr = key->GetValueTrack();
  MS_EXCEPTION_IF_NULL(key_value_ptr);
  auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
  if (key_value_track == nullptr) {
    MS_LOG(EXCEPTION) << "EnvironSet evaluator args[1] expected should be able to cast to SymbolicKeyInstancePtrbut: "
                      << key_value_ptr->ToString();
  }
  auto expected = key_value_track->abstract();
  MS_EXCEPTION_IF_NULL(expected);

  auto value = args_spec_list[kIndexTwo];
  MS_LOG(DEBUG) << "key: " << key->ToString() << ", value: " << value->ToString();
  if (value->isa<AbstractUndetermined>() && !value->isa<AbstractTensor>()) {
    EnvSetSparseResultMgr::GetInstance().Set(true);
  }
  return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
}

AbstractBasePtr InferImplEnvironAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // args: Three objects of a subclass of AbstractBase, env, key, dflt(default).
  constexpr auto environ_add_input_size = 2;
  CheckArgsSize(primitive->name(), args_spec_list, environ_add_input_size);
  return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
}

AbstractBasePtr InferImplEnvironDestroyAll(const AnalysisEnginePtr &, const PrimitivePtr &,
                                           const AbstractBasePtrList &) {
  return std::make_shared<abstract::AbstractScalar>(kAnyValue, std::make_shared<Bool>());
}

AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // args: Two objects of a subclass of AbstractBase, key and value.
  constexpr auto state_setitem_input_size = 2;
  CheckArgsSize(primitive->name(), args_spec_list, state_setitem_input_size);

  TypePtr type = args_spec_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kObjectTypeRefKey && type->type_id() != kObjectTypeSymbolicKeyType) {
    MS_LOG(EXCEPTION) << "First input of StateSetItem should be a RefKey or SymbolicKeyType but a " << type->ToString();
  }
  return std::make_shared<AbstractScalar>(kAnyValue, kBool);
}

AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  constexpr auto depend_input_size = 2;
  CheckArgsSize(primitive->name(), args_spec_list, depend_input_size);

  // If the dependent has a value, just return depended node.
  // If depended node is not Any, the dependent maybe eliminated.
  auto dependant_abstract = args_spec_list[1];
  auto dependant_value = dependant_abstract->BuildValue();
  MS_EXCEPTION_IF_NULL(dependant_value);
  if (dependant_value != kAnyValue) {
    return args_spec_list[0];
  }
  auto depends = args_spec_list[0];

  if (depends->isa<AbstractRefTensor>()) {
    auto abs_ref = depends->cast<AbstractRefPtr>();
    auto tensor_abs = abs_ref->ref();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    return std::make_shared<AbstractRefTensor>(tensor_abs->Broaden()->cast<AbstractTensorPtr>(),
                                               abs_ref->ref_key_value());
  }

  auto depends_abs = depends->Broaden();  // Avoid eliminating the dependent node.
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR)) {
    // For scalar, need to set value to kAnyValue, because broaden scalar will not change the value.
    if (depends_abs->isa<AbstractScalar>()) {
      depends_abs->set_value(kAnyValue);
    }
  }
  return depends_abs;
}

AbstractBasePtr InferImplUpdateState(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << primitive->name() << " input args size should be at least 1, but got 0";
  }
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  constexpr size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto values = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dense_shape = CheckArg<AbstractTuple>(op_name, args_spec_list, 2);

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

AbstractBasePtr InferImplRowTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: row tensor and tensor.
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_size);
  auto row_tensor = CheckArg<AbstractRowTensor>(op_name, args_spec_list, 0);
  auto tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(row_tensor->dense_shape());
  MS_EXCEPTION_IF_NULL(tensor->shape());
  return args_spec_list[0];
}

AbstractBasePtr InferImplAllSwap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  constexpr auto all_swap_input_size = 3;
  CheckArgsSize(op_name, args_spec_list, all_swap_input_size);
  auto tensor_in = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(tensor_in);
  MS_EXCEPTION_IF_NULL(tensor_in->shape());
  auto tensor_in_shape = tensor_in->shape()->shape();

  auto send_size = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(send_size);
  auto recv_size = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  MS_EXCEPTION_IF_NULL(recv_size);

  // Get the content of the recv size
  auto recv_size_value_ptr = recv_size->BuildValue();
  MS_EXCEPTION_IF_NULL(recv_size_value_ptr);
  auto recv_size_tensor = recv_size_value_ptr->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(recv_size_tensor);
  auto data_pos = static_cast<int64_t *>(recv_size_tensor->data_c());
  MS_EXCEPTION_IF_NULL(data_pos);
  int64_t infer_max_size = 0;
  for (size_t i = 0; i < recv_size_tensor->DataSize(); ++i) {
    infer_max_size += *(data_pos + i);
  }

  ShapeVector tensor_out_shape = {Shape::kShapeDimAny, tensor_in_shape[1]};
  ShapeVector max_shape = {infer_max_size / tensor_in_shape[1], tensor_in_shape[1]};
  auto tensor_out =
    std::make_shared<AbstractTensor>(tensor_in->element(), std::make_shared<Shape>(tensor_out_shape, max_shape));

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(tensor_out->element(), std::make_shared<Shape>(tensor_out_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplAllReduce(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(x->shape()->shape()));
}

AbstractBasePtr InferImplBroadcast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(x->shape()->shape()));
}

AbstractBasePtr InferImplAllGather(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto tmp_shape = x->shape()->shape();
  if (!primitive->HasAttr(kRankSize)) {
    MS_LOG(EXCEPTION) << "Primitive don't have rank_size attr";
  }
  auto rank_size = GetValue<int>(primitive->GetAttr(kRankSize));
  if (rank_size == 0) {
    MS_LOG(EXCEPTION) << "rank_size is 0";
  }
  if (tmp_shape.empty()) {
    MS_LOG(EXCEPTION) << "shape size is 0";
  }
  if (tmp_shape[0] > 0) {
    tmp_shape[0] = tmp_shape[0] * rank_size;
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(tmp_shape));
}

AbstractBasePtr InferImplReduceScatter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto tmp_shape = x->shape()->shape();
  if (!primitive->HasAttr(kRankSize)) {
    MS_LOG(EXCEPTION) << "Primitive don't have rank_size attr";
  }
  auto rank_size = GetValue<int>(primitive->GetAttr(kRankSize));
  if (tmp_shape.empty()) {
    MS_LOG(EXCEPTION) << "shape size is 0";
  }
  tmp_shape[0] = LongMulWithOverflowCheck(tmp_shape[0], rank_size);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(tmp_shape));
}

AbstractBasePtr InferImplMemCpyAsync(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(x->shape()->shape()));
}

AbstractBasePtr InferImplCast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  // GPU has 2 inputs while tbe has 1 only. Skip CheckArgsSize.
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input_x);

  ValuePtr dst_type;
  constexpr auto kCastInputSize = 2;
  if (args_spec_list.size() < kCastInputSize) {
    dst_type = primitive->GetAttr("dst_type");
  } else {
    auto type_abs = CheckArg<AbstractType>(op_name, args_spec_list, 1);
    dst_type = type_abs->BuildValue();
  }

  MS_EXCEPTION_IF_NULL(dst_type);
  if (!dst_type->isa<Type>()) {
    MS_LOG(EXCEPTION) << "Invalid Cast dst_type " << dst_type->ToString();
  }
  auto input_type = dst_type->cast<TypePtr>();
  auto ret = std::make_shared<AbstractTensor>(input_type, input_x->shape());
  return ret;
}

AbstractBasePtr InferImplIsDimUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, input_size);
  auto abs = args_spec_list[0];
  if (!abs->isa<AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "The input of " << op_name << " should be tuple but got " << abs->ToString();
  }
  auto abs_seq = abs->cast<AbstractSequencePtr>();
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(abs_seq->dynamic_len()), kBool);
}

AbstractBasePtr InferImplIsShapeUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, input_size);
  auto abs = args_spec_list[0];
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
      if (cur_val == kAnyValue) {
        is_shape_unknown = true;
        break;
      }
    }
  }
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(is_shape_unknown), kBool);
}

AbstractBasePtr InferImplIsElementUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  constexpr size_t input_size = 1;
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, input_size);
  auto abs = args_spec_list[0];
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
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: Ref/Tensor, universal
  constexpr auto load_input_size = 2;
  CheckArgsSize(primitive->name(), args_spec_list, load_input_size);
  auto ref_abs = dyn_cast<abstract::AbstractRefTensor>(args_spec_list[0]);
  if (ref_abs != nullptr) {
    // Return tensor value if input is Ref.
    return ref_abs->CloneAsTensor();
  }
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplTransData(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  auto output = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(output);
  return output;
}
AbstractBasePtr InferImplAdamApplyOne(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  constexpr auto adam_input_size = 10;
  CheckArgsSize(primitive->name(), args_spec_list, adam_input_size);
  auto input0 = args_spec_list[0];
  auto input1 = args_spec_list[1];
  auto input2 = args_spec_list[2];
  auto input3 = args_spec_list[3];
  auto input4 = args_spec_list[4];
  auto mul0_x = args_spec_list[5];
  auto mul1_x = args_spec_list[6];
  auto mul2_x = args_spec_list[7];
  auto mul3_x = args_spec_list[8];
  auto add2_y = args_spec_list[9];

  auto square0 = ops::SquareInfer(nullptr, primitive, {input0});
  auto mul1 = ops::MulInfer(nullptr, primitive, {mul1_x, input0});
  auto mul0 = ops::MulInfer(nullptr, primitive, {mul0_x, input2});
  auto mul2 = ops::MulInfer(nullptr, primitive, {mul2_x, input1});
  auto mul3 = ops::MulInfer(nullptr, primitive, {mul3_x, square0});
  auto add0 = ops::AddInfer(nullptr, primitive, {mul0, mul1});
  auto add1 = ops::AddInfer(nullptr, primitive, {mul2, mul3});
  auto sqrt0 = InferImplSqrt(nullptr, primitive, {add1});
  auto add2 = ops::AddInfer(nullptr, primitive, {add2_y, sqrt0});
  auto true_div0 = ops::RealDivInfer(nullptr, primitive, {add0, add2});
  auto mul4 = ops::MulInfer(nullptr, primitive, {input4, true_div0});
  auto sub0 = ops::SubInfer(nullptr, primitive, {input3, mul4});

  AbstractBasePtrList rets = {add1, add0, sub0};
  return std::make_shared<AbstractTuple>(rets);
}
AbstractBasePtr InferImplTensorMove(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  auto output = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(output);
  return output;
}
AbstractBasePtr InferImplAdamApplyOneWithDecay(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list) {
  // An object of a subclass of AbstractBase
  constexpr auto adam_input_size = 11;
  CheckArgsSize(primitive->name(), args_spec_list, adam_input_size);
  auto input0 = args_spec_list[0];
  auto input1 = args_spec_list[1];
  auto input2 = args_spec_list[2];
  auto input3 = args_spec_list[3];
  auto input4 = args_spec_list[4];
  auto mul0_x = args_spec_list[5];
  auto mul1_x = args_spec_list[6];
  auto mul2_x = args_spec_list[7];
  auto mul3_x = args_spec_list[8];
  auto mul4_x = args_spec_list[9];
  auto add2_y = args_spec_list[10];

  auto mul0 = ops::MulInfer(nullptr, primitive, {mul0_x, input2});
  auto mul1 = ops::MulInfer(nullptr, primitive, {mul1_x, input0});
  auto square0 = ops::SquareInfer(nullptr, primitive, {input0});
  auto add0 = ops::AddInfer(nullptr, primitive, {mul0, mul1});
  auto mul2 = ops::MulInfer(nullptr, primitive, {mul2_x, input1});
  auto mul3 = ops::MulInfer(nullptr, primitive, {mul3_x, square0});
  auto add1 = ops::AddInfer(nullptr, primitive, {mul2, mul3});
  auto sqrt0 = InferImplSqrt(nullptr, primitive, {add1});
  auto add2 = ops::AddInfer(nullptr, primitive, {add2_y, sqrt0});
  auto mul4 = ops::MulInfer(nullptr, primitive, {mul4_x, input3});
  auto real_div0 = ops::RealDivInfer(nullptr, primitive, {add0, add2});
  auto add3 = ops::AddInfer(nullptr, primitive, {mul4, real_div0});
  auto mul5 = ops::MulInfer(nullptr, primitive, {input4, add3});
  auto sub0 = ops::SubInfer(nullptr, primitive, {input3, mul5});
  AbstractBasePtrList rets = {add1, add0, sub0};
  return std::make_shared<AbstractTuple>(rets);
}
// Infer for MapTensor.default_value.
AbstractBasePtr InferImplMapTensorGetDefaultValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  const auto &arg = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->default_value());
}
// Infer for MapTensor.permit_filter_value.
AbstractBasePtr InferImplMapTensorGetPermitFilterValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                       const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  const auto &arg = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->permit_filter_value());
}
// Infer for MapTensor.evict_filter_value.
AbstractBasePtr InferImplMapTensorGetEvictFilterValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                      const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  const auto &arg = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(arg);
  auto abs_map_tensor = arg->cast_ptr<abstract::AbstractMapTensor>();
  if (abs_map_tensor == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect MapTensor, but got " << arg->ToString();
  }
  return std::make_shared<AbstractScalar>(abs_map_tensor->evict_filter_value());
}
}  // namespace abstract
}  // namespace mindspore
