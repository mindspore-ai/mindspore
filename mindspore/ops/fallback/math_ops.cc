/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/matmul_ext.h"
#include "ops_utils/op_utils.h"
#include "include/op_enum.h"

namespace mindspore {
namespace expander {
namespace {
const std::set<TypeId> kIntergralSet = {kNumberTypeBool,  kNumberTypeUInt8, kNumberTypeInt8,
                                        kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};

const std::set<TypeId> kFloatSet = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};

NodePtr Expand(FallbackIRBuilder *ib, NodePtr tensor, size_t ndim) {
  ShapeVector shape = tensor->shape();
  while (shape.size() < ndim) {
    shape.insert(shape.begin(), 1);
  }
  tensor = ib->Reshape(tensor, ib->Value(shape));
  return tensor;
}
}  // namespace
REG_FALLBACK_BUILDER("AddExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  if (x->dtype()->type_id() == kNumberTypeBool) {
    return {};
  }
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x + y * alpha_tensor};
});

REG_FALLBACK_BUILDER("AddScalar").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);

  auto x_type = ib->GetDtype(x)->type_id();
  auto y_type = ib->GetDtype(y)->type_id();
  if ((y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64) &&
      (x_type == kNumberTypeUInt16 || x_type == kNumberTypeUInt32 || x_type == kNumberTypeUInt64)) {
    MS_EXCEPTION(TypeError) << "Type implicit conversion between Tensor[" << TypeIdToString(x_type) << "] and "
                            << TypeIdToString(y_type) << " is not supported.";
  }

  std::set<TypeId> kSet = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  auto promote_type = TypeIdToType(kNumberTypeFloat32);
  if ((kSet.find(x_type) != kSet.end()) && y_type == kNumberTypeFloat32) {
    promote_type = TypeIdToType(kNumberTypeFloat32);
  } else if (x_type == kNumberTypeBool && (y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64)) {
    promote_type = TypeIdToType(y_type);
  } else {
    promote_type = TypeIdToType(x_type);
  }
  auto x_cast = ib->Cast(x, promote_type);
  auto y_cast = ib->ScalarToTensor(y, promote_type);
  auto alpha_cast = ib->ScalarToTensor(alpha, promote_type);

  return {ib->Add(x_cast, ib->Mul(y_cast, alpha_cast))};
});

REG_FALLBACK_BUILDER("SubScalar").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);

  auto x_type = ib->GetDtype(x)->type_id();
  auto y_type = ib->GetDtype(y)->type_id();
  if ((y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64) &&
      (x_type == kNumberTypeUInt16 || x_type == kNumberTypeUInt32 || x_type == kNumberTypeUInt64)) {
    MS_EXCEPTION(TypeError) << "Type implicit conversion between Tensor[" << TypeIdToString(x_type) << "] and "
                            << TypeIdToString(y_type) << " is not supported.";
  }

  std::set<TypeId> kSet = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  auto promote_type = TypeIdToType(kNumberTypeFloat32);
  if ((kSet.find(x_type) != kSet.end()) && y_type == kNumberTypeFloat32) {
    promote_type = TypeIdToType(kNumberTypeFloat32);
  } else if (x_type == kNumberTypeBool && (y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64)) {
    promote_type = TypeIdToType(y_type);
  } else {
    promote_type = TypeIdToType(x_type);
  }
  auto x_cast = ib->Cast(x, promote_type);
  auto y_cast = ib->ScalarToTensor(y, promote_type);
  auto alpha_cast = ib->ScalarToTensor(alpha, promote_type);

  return {ib->Sub(x_cast, ib->Mul(y_cast, alpha_cast))};
});

REG_FALLBACK_BUILDER("GreaterEqualScalar").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);

  auto x_type = ib->GetDtype(x)->type_id();
  auto y_type = ib->GetDtype(y)->type_id();
  if ((y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64) &&
      (x_type == kNumberTypeUInt16 || x_type == kNumberTypeUInt32 || x_type == kNumberTypeUInt64)) {
    MS_EXCEPTION(TypeError) << "Type implicit conversion between Tensor[" << TypeIdToString(x_type) << "] and "
                            << TypeIdToString(y_type) << " is not supported.";
  }

  std::set<TypeId> kSet = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  auto promote_type = TypeIdToType(kNumberTypeFloat32);
  if ((kSet.find(x_type) != kSet.end()) && y_type == kNumberTypeFloat32) {
    promote_type = TypeIdToType(kNumberTypeFloat32);
  } else if (x_type == kNumberTypeBool && (y_type == kNumberTypeFloat32 || y_type == kNumberTypeInt64)) {
    promote_type = TypeIdToType(y_type);
  } else {
    promote_type = TypeIdToType(x_type);
  }
  auto x_cast = ib->Cast(x, promote_type);
  auto y_cast = ib->ScalarToTensor(y, promote_type);

  return {ib->GreaterEqual(x_cast, y_cast)};
});

REG_FALLBACK_BUILDER("InplaceAddExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->ScalarToTensor(alpha, x->dtype());
  auto y_cast = ib->Cast(y, x->dtype());
  return {x + y_cast * alpha_tensor};
});

REG_FALLBACK_BUILDER("InplaceAddsExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  auto alpha_tensor = ib->ScalarToTensor(alpha, x->dtype());
  return {x + other_tensor * alpha_tensor};
});

REG_FALLBACK_BUILDER("InplaceSubExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->ScalarToTensor(alpha, x->dtype());
  auto y_cast = ib->Cast(y, x->dtype());
  return {x - y_cast * alpha_tensor};
});

REG_FALLBACK_BUILDER("InplaceSubScalar").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  auto alpha_tensor = ib->ScalarToTensor(alpha, x->dtype());
  return {x - other_tensor * alpha_tensor};
});

REG_FALLBACK_BUILDER("InplaceMul").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto y_cast = ib->Cast(y, x->dtype());
  return {x * y_cast};
});

REG_FALLBACK_BUILDER("InplaceMuls").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  return {x * other_tensor};
});

REG_FALLBACK_BUILDER("InplaceDiv").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto y_cast = ib->Cast(y, x->dtype());
  return {x / y_cast};
});

REG_FALLBACK_BUILDER("InplaceDivs").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  return {x / other_tensor};
});

REG_FALLBACK_BUILDER("Divs").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  return {x / other_tensor};
});

REG_FALLBACK_BUILDER("InplaceFloorDivide").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto y_cast = ib->Cast(y, x->dtype());
  return {ib->Emit("FloorDiv", {x, y_cast})};
});

REG_FALLBACK_BUILDER("InplaceFloorDivides").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto other_tensor = ib->ScalarToTensor(y, x->dtype());
  return {ib->Emit("FloorDiv", {x, other_tensor})};
});

REG_FALLBACK_BUILDER("SubExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  if (x->dtype()->type_id() == kNumberTypeBool) {
    return {};
  }
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x - y * alpha_tensor};
});

REG_FALLBACK_BUILDER("Muls").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto x_type = ib->GetDtype(x)->type_id();
  auto y_type = ib->GetDtype(y)->type_id();
  TypeId promote_type_id = ops::PromoteTensorScalarType(x_type, y_type);
  auto promote_type = TypeIdToType(promote_type_id);
  MS_EXCEPTION_IF_NULL(promote_type);
  auto x_cast = ib->Cast(x, promote_type);
  auto y_cast = ib->ScalarToTensor(y, promote_type);
  return {ib->Mul(x_cast, y_cast)};
});

REG_FALLBACK_BUILDER("BatchMatMulExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->Cast(ib->BatchMatMul(x, y, false, false), ib->GetDtype(x))};
});

DEF_PURE_SHAPE_CALC(g_matmul_ext_fallback_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &input_shape = inputs.at(kIndex0);
    auto &weight_shape = inputs.at(kIndex1);

    bool is_weight_scalar = weight_shape.size() == 1;

    ShapeVector multiplication_shape = ops::CheckMatMulShapes(input_shape, weight_shape);
    ShapeVector broadcast_shape_input = ops::GetMatMulExtBroadcastShape(multiplication_shape, input_shape);
    ShapeVector broadcast_shape_weight = ops::GetMatMulExtBroadcastShape(multiplication_shape, weight_shape);
    ShapeVector output_shape = ops::InferShapeRem(multiplication_shape, input_shape, weight_shape, is_weight_scalar);
    ShapeVector transpose_order;
    size_t max_dim_count = multiplication_shape.size() + 2;

    for (size_t i = 0; i < max_dim_count; ++i) {
      transpose_order.push_back(i);
    }

    int64_t total_batch_size = 1;
    for (auto dim_size : multiplication_shape) {
      total_batch_size *= dim_size;
    }

    ShapeVector final_input_shape = {total_batch_size, broadcast_shape_input[broadcast_shape_input.size() - 2],
                                     broadcast_shape_input[broadcast_shape_input.size() - 1]};
    ShapeVector final_weight_shape = {total_batch_size, broadcast_shape_weight[broadcast_shape_weight.size() - 2],
                                      broadcast_shape_weight[broadcast_shape_weight.size() - 1]};

    if (is_weight_scalar) {
      std::swap(transpose_order[max_dim_count - 1], transpose_order[max_dim_count - 2]);
      std::swap(final_weight_shape[final_weight_shape.size() - 1], final_weight_shape[final_weight_shape.size() - 2]);
    }

    return {broadcast_shape_input, broadcast_shape_weight, transpose_order,
            final_input_shape,     final_weight_shape,     output_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    int64_t broadcast_rank_input = -1LL;
    int64_t broadcast_rank_weight = -1LL;
    int64_t transpose_order_rank = -1LL;
    int64_t final_input_shape_rank = -1LL;
    int64_t final_weight_shape_rank = -1LL;
    int64_t output_shape_rank = -1LL;

    if (!IsDynamicRank(inputs[0]) && !IsDynamicRank(inputs[1])) {
      auto &input_shape = inputs.at(kIndex0);
      auto &weight_shape = inputs.at(kIndex1);

      size_t max_dim_count = std::max(input_shape.size(), weight_shape.size());
      max_dim_count = std::max(max_dim_count, static_cast<size_t>(2));

      if (input_shape.size() == 1 && weight_shape.size() == 1) {
        output_shape_rank = 0;
      } else if (input_shape.size() == 1 || weight_shape.size() == 1) {
        output_shape_rank = max_dim_count - 1;
      } else {
        output_shape_rank = max_dim_count;
      }

      broadcast_rank_input = broadcast_rank_weight = transpose_order_rank = max_dim_count;
      final_input_shape_rank = final_weight_shape_rank = 3;
    }
    return {broadcast_rank_input,   broadcast_rank_weight,   transpose_order_rank,
            final_input_shape_rank, final_weight_shape_rank, output_shape_rank};
  });

REG_FALLBACK_BUILDER("MatMulExt").SetBody(BODYFUNC(ib) {
  NodePtr input = ib->GetInput(kIndex0);
  NodePtr other = ib->GetInput(kIndex1);
  if (IsDynamic(input->shape()) || IsDynamic(other->shape())) {
    auto shapes = ib->ShapeCalc(g_matmul_ext_fallback_shapecalc, {input, other});
    input = ib->Emit("BroadcastTo", {input, shapes[0]});
    other = ib->Emit("BroadcastTo", {other, shapes[1]});
    other = ib->Transpose(other, shapes[2]);
    input = ib->Reshape(input, shapes[3]);
    other = ib->Reshape(other, shapes[4]);
    auto ret = ib->Cast(ib->BatchMatMul(input, other), ib->GetDtype(input));
    ret = ib->Reshape(ret, shapes[5]);
    return {ret};
  } else {
    const ShapeVector shape1_orig = input->shape();
    const ShapeVector shape2_orig = other->shape();
    auto output_dtype = ib->GetDtype(input);
    auto shape_backbone = ops::CheckMatMulShapes(shape1_orig, shape2_orig);
    bool is_empty_tensor =
      std::any_of(shape1_orig.begin(), shape1_orig.end(), [](const auto &element) { return element == 0; });
    if (is_empty_tensor) {
      return {ib->Tensor(0, input->dtype())};
    }
    auto input_rank = input->shape().size();
    auto other_rank = other->shape().size();
    bool transpose_b = other_rank == 1;
    auto shape_out = ops::InferShapeRem(shape_backbone, shape1_orig, shape2_orig, transpose_b);
    input = input_rank == 1 ? Expand(ib, input, 2) : input;
    other = other_rank == 1 ? Expand(ib, other, 2) : other;
    if (input->shape().size() == 2 && other->shape().size() == 2) {
      auto ret = ib->MatMul(input, other, false, transpose_b);
      ret = ib->Cast(ib->Reshape(ret, shape_out), output_dtype);
      return {ret};
    }
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (target == kAscendDevice) {
      auto ret = ib->BatchMatMul(input, other, false, transpose_b);
      ret = ib->Cast(ib->Reshape(ret, shape_out), output_dtype);
      return {ret};
    }
    ShapeVector broadcast_shape1 = ops::GetMatMulExtBroadcastShape(shape_backbone, shape1_orig);
    ShapeVector broadcast_shape2 = ops::GetMatMulExtBroadcastShape(shape_backbone, shape2_orig);
    if (input->shape() != broadcast_shape1) {
      input = ib->Emit("BroadcastTo", {input, ib->Value(broadcast_shape1)});
    }
    if (other->shape() != broadcast_shape2) {
      other = ib->Emit("BroadcastTo", {other, ib->Value(broadcast_shape2)});
    }
    auto ret = ib->BatchMatMul(input, other, false, transpose_b);
    ret = ib->Cast(ib->Reshape(ret, shape_out), output_dtype);
    return {ret};
  }
});

REG_FALLBACK_BUILDER("MeanExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dtype = ib->GetInput(kIndex3);

  auto dtype_type = dtype->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_type);
  // cppcheck-suppress *
  if (!dtype_type->isa<TypeNone>()) {
    auto dtype_opt = GetScalarValue<int64_t>(dtype->BuildValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), "For 'MeanExt', dtype must have valid value.");
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  }

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  auto out = ib->Emit("ReduceMean", {input, axis, keep_dims});
  return {out};
});

REG_FALLBACK_BUILDER("SumExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dtype = ib->GetInput(kIndex3);

  auto dtype_type = dtype->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_type);
  if (!dtype_type->isa<TypeNone>()) {
    auto dtype_opt = GetScalarValue<int64_t>(dtype->BuildValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), "For 'SumExt', dtype must have valid value.");
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  } else {
    auto input_type = input->dtype()->type_id();
    if (kIntergralSet.find(input_type) != kIntergralSet.end()) {
      input = ib->Cast(input, kInt64);
    }
  }

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  auto out = ib->Emit("ReduceSum", {input, axis, keep_dims, ib->Value<bool>(false)});
  return {out};
});

REG_FALLBACK_BUILDER("ProdExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dtype = ib->GetInput(kIndex3);

  MS_LOG(DEBUG) << "Fallback Expander 'ProdExt' start";

  if (dtype->abstract()->BuildType()->isa<TypeNone>()) {
    auto input_type = input->dtype()->type_id();
    if (kIntergralSet.find(input_type) != kIntergralSet.end()) {
      input = ib->Cast(input, kInt64);
    }
  } else {
    auto dtype_opt = GetScalarValue<int64_t>(dtype->BuildValue());
    if (!dtype_opt.has_value()) {
      MS_LOG(EXCEPTION) << "For 'ProdExt', dtype must have valid value.";
    }
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  }

  const auto axis_abs = axis->abstract();
  if (axis_abs->BuildType()->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  } else if (axis_abs->isa<abstract::AbstractScalar>()) {
    axis = ib->MakeTuple({axis});
  } else if (axis_abs->isa<abstract::AbstractTensor>()) {
    axis = ib->TensorToTuple({axis});
  } else {
    MS_LOG(EXCEPTION) << "For 'ProdExt', axis got an unexpected type: " << axis->abstract();
  }

  auto out = ib->Emit("ReduceProd", {input, axis, keep_dims});
  return {out};
});

REG_FALLBACK_BUILDER("TraceV2").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto offset = ib->GetInput(kIndex1);
  auto axis1 = ib->GetInput(kIndex2);
  auto axis2 = ib->GetInput(kIndex3);
  auto dtype = ib->GetInput(kIndex4);
  auto diag = ib->Emit("Diagonal", {x, offset, axis1, axis2});
  NodePtr casted_diag;
  if (dtype->BuildValue()->type_name() != "None") {
    int64_t dtype_value = GetValue<int64_t>(dtype->BuildValue());
    TypeId dtype_id = static_cast<TypeId>(dtype_value);
    casted_diag = ib->Cast(diag, TypeIdToType(dtype_id));
  } else {
    auto input_type = diag->GetType();
    auto input_type_ptr = input_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(input_type_ptr);
    auto input_type_id = input_type_ptr->element()->type_id();
    static const std::vector<TypeId> type_to_int64 = {
      kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt16, kNumberTypeUInt32,
    };
    bool is_type_to_int64 = std::any_of(type_to_int64.begin(), type_to_int64.end(),
                                        [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
    if (is_type_to_int64) {
      casted_diag = ib->Cast(diag, kInt64);
    } else {
      casted_diag = diag;
    }
  }
  auto out = ib->Emit(
    "ReduceSum", {casted_diag, ib->Value<std::vector<int64_t>>({-1}), ib->Value<bool>(false), ib->Value<bool>(false)});
  return {out};
});

NodePtr BuilderForMaxorMin(FallbackIRBuilder *ib, const std::string &emit_op) {
  auto input = ib->GetInput(kIndex0);
  // empty axis: all dimensions will be reduced
  std::vector<int64_t> axis;
  auto input_shape = input->shape();
  // The GE backend may be used under static shape and the empty axis needs to be expanded to represent
  // that all dimensions will be reduced.
  if (!IsDynamic(input_shape)) {
    auto input_shape_len = SizeToLong(input_shape.size());
    for (int64_t i = 0; i < input_shape_len; ++i) {
      axis.push_back(i);
    }
  }
  auto axis_value = ib->Value(axis);
  auto keep_dims = ib->Value(false);
  auto out = ib->Emit(emit_op, {input, axis_value, keep_dims});
  return out;
}

REG_FALLBACK_BUILDER("Max").SetBody(BODYFUNC(ib) { return {BuilderForMaxorMin(ib, "ReduceMax")}; });

REG_FALLBACK_BUILDER("Min").SetBody(BODYFUNC(ib) { return {BuilderForMaxorMin(ib, "ReduceMin")}; });

REG_FALLBACK_BUILDER("DivMod").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto rounding_mode = ib->GetInput(kIndex2);

  auto mode_type = rounding_mode->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(mode_type);
  if (mode_type->isa<TypeNone>()) {
    return {ib->Div(input_x, input_y)};
  }

  auto mode_value_ptr = rounding_mode->BuildValue();
  auto mode_opt = mindspore::GetScalarValue<int64_t>(mode_value_ptr);

  if (mode_opt.value() == ops::RoundingMode::FLOOR) {
    return {ib->Emit("FloorDiv", {input_x, input_y})};
  } else if (mode_opt.value() == ops::RoundingMode::TRUNC) {
    auto div_out = ib->Cast(ib->Div(input_x, input_y), ib->GetDtype(input_x)->type_id());
    return {ib->Emit("Trunc", {div_out})};
  } else {
    MS_LOG(EXCEPTION) << "DivMod abstract failed.";
  }
});

REG_FALLBACK_BUILDER("InplaceRemainderTensorTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto other_cast = ib->Cast(other, input->dtype());
  return {ib->Emit("FloorMod", {input, other_cast})};
});

REG_FALLBACK_BUILDER("InplaceRemainderTensorScalar").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto other_cast = ib->ScalarToTensor(other, input->dtype());
  return {ib->Emit("FloorMod", {input, other_cast})};
});

REG_FALLBACK_BUILDER("RemainderTensorTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto promote_type = mindspore::ops::PromoteType(ib->GetDtype(input), ib->GetDtype(other), "RemainderTensorTensor");
  MS_EXCEPTION_IF_NULL(promote_type);
  auto input_cast = ib->Cast(input, promote_type);
  auto other_cast = ib->Cast(other, promote_type);
  return {ib->Emit("FloorMod", {input_cast, other_cast})};
});

REG_FALLBACK_BUILDER("RemainderTensorScalar").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto input_type = ib->GetDtype(input);
  auto promote_type = input_type;
  if (input_type->type_id() == kNumberTypeBool) {
    promote_type = other->dtype();
  } else if (kIntergralSet.find(input_type->type_id()) != kIntergralSet.end() &&
             kFloatSet.find(other->dtype()->type_id()) != kFloatSet.end()) {
    promote_type = TypeIdToType(kNumberTypeFloat32);
  }
  MS_EXCEPTION_IF_NULL(promote_type);
  auto input_cast = ib->Cast(input, promote_type);
  auto other_cast = ib->ScalarToTensor(other, promote_type);
  return {ib->Emit("FloorMod", {input_cast, other_cast})};
});

REG_FALLBACK_BUILDER("RemainderScalarTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto other_type = ib->GetDtype(other);
  auto promote_type = other_type;
  if (other_type->type_id() == kNumberTypeBool) {
    promote_type = input->dtype();
  } else if (kIntergralSet.find(other_type->type_id()) != kIntergralSet.end() &&
             kFloatSet.find(input->dtype()->type_id()) != kFloatSet.end()) {
    promote_type = TypeIdToType(kNumberTypeFloat32);
  }
  MS_EXCEPTION_IF_NULL(promote_type);
  auto input_cast = ib->ScalarToTensor(input, promote_type);
  auto other_cast = ib->Cast(other, promote_type);
  return {ib->Emit("FloorMod", {input_cast, other_cast})};
});

REG_FALLBACK_BUILDER("EqualCount").SetBody(BODYFUNC(ib) {
  // Check inputs
  const auto &input_x = ib->GetInput(kIndex0);
  const auto &input_y = ib->GetInput(kIndex1);
  if (input_x->dtype()->type_id() != input_y->dtype()->type_id()) {
    MS_LOG(WARNING) << "In EqualCount, two inputs should have same data type, but type of input_x is "
                    << input_x->dtype()->ToString() << ", type of input_y is " << input_y->dtype()->ToString();
    return {};
  }

  if (input_x->dtype()->type_id() != kNumberTypeFloat32 && input_x->dtype()->type_id() != kNumberTypeFloat16 &&
      input_x->dtype()->type_id() != kNumberTypeInt32) {
    MS_LOG(WARNING) << "In EqualCount, dtype of inputs must be float16 or float32 or int32, but get data type:"
                    << input_x->dtype()->ToString();
    return {};
  }

  if (input_x->shape() != input_y->shape()) {
    MS_LOG(WARNING) << "In EqualCount, two inputs should have same shape, but shape of input_x is " << input_x->shape()
                    << ", shape of input_y is " << input_y->shape();
    return {};
  }
  // Expand
  auto dtype = input_x->dtype();
  auto eql_val = ib->Equal(input_x, input_y);
  auto cast_val = ib->Cast(eql_val, kNumberTypeFloat32);
  auto shape_size = input_x->shape().size();
  std::vector<int64_t> axis(shape_size);
  for (size_t i = 0; i < shape_size; ++i) {
    axis[i] = SizeToLong(i);
  }
  auto result = ib->ReduceSum(cast_val, axis, false);
  result = ib->Reshape(result, {1});
  if (result->dtype() != dtype) {
    result = ib->Cast(result, dtype->type_id());
  }
  return {result};
});

REG_FALLBACK_BUILDER("PowTensorScalar").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto exponent = ib->GetInput(kIndex1);

  auto input_type = ib->GetDtype(input)->type_id();
  auto exponent_type = ib->GetDtype(exponent)->type_id();
  TypeId promote_type_id = ops::PromoteTensorScalarType(input_type, exponent_type);
  auto promote_type = TypeIdToType(promote_type_id);
  MS_EXCEPTION_IF_NULL(promote_type);
  auto input_cast = ib->Cast(input, promote_type);
  auto exp_tensor = ib->ScalarToTensor(exponent, promote_type);
  auto out = ib->Pow(input_cast, exp_tensor);
  return {out};
});

REG_FALLBACK_BUILDER("PowScalarTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto exponent = ib->GetInput(kIndex1);

  auto exponent_type = ib->GetDtype(exponent)->type_id();
  auto input_type = ib->GetDtype(input)->type_id();
  TypeId promote_type_id = ops::PromoteTensorScalarType(exponent_type, input_type);
  auto promote_type = TypeIdToType(promote_type_id);
  MS_EXCEPTION_IF_NULL(promote_type);
  auto input_tensor = ib->ScalarToTensor(input, promote_type);
  auto exponent_cast = ib->Cast(exponent, promote_type);
  auto out = ib->Pow(input_tensor, exponent_cast);
  return {out};
});

REG_FALLBACK_BUILDER("BitwiseAndTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto other = ib->GetInput(kIndex1);
  auto input_type = input->dtype()->type_id();
  NodePtr out{nullptr};
  if (input_type == kNumberTypeBool) {
    auto int_input = ib->Cast(input, kInt8);
    auto int_other = ib->Cast(other, kInt8);
    out = ib->Cast(ib->Emit("BitwiseAnd", {int_input, int_other}), input_type);
  } else {
    out = ib->Emit("BitwiseAnd", {input, other});
  }
  return {out};
});
}  // namespace expander
}  // namespace mindspore
