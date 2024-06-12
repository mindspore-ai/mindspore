/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/matmul_ext.h"
#include "ops/op_utils.h"
#include "ops/op_enum.h"

namespace mindspore {
namespace expander {
namespace {
const std::set<TypeId> kIntergralSet = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                        kNumberTypeInt32};

size_t Rank(NodePtr x) {
  if (IsDynamicRank(x->shape())) {
    MS_LOG(EXCEPTION) << "Rank of input shape is dynamic.";
  }

  return x->shape().size();
}

NodePtr Expand(FallbackIRBuilder *ib, NodePtr tensor, size_t ndim) {
  ShapeVector shape = tensor->shape();
  while (shape.size() < ndim) {
    shape.insert(shape.begin(), 1);
  }
  tensor = ib->Reshape(tensor, ib->Value(shape));
  return tensor;
}

ShapeVector ReduceTo3D(const ShapeVector &shape) {
  ShapeVector ret;

  int64_t dim0 = 1;
  for (size_t i = 0; i < shape.size() - kDim2; ++i) {
    dim0 *= shape[i];
  }
  ret.push_back(dim0);
  ret.push_back(shape[shape.size() - kDim2]);
  ret.push_back(shape[shape.size() - kDim1]);
  return ret;
}
}  // namespace
REG_FALLBACK_BUILDER("AddExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x + y * alpha_tensor};
});

REG_FALLBACK_BUILDER("SubExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x - y * alpha_tensor};
});

REG_FALLBACK_BUILDER("BatchMatMulExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->BatchMatMul(x, y, false, false)};
});

REG_FALLBACK_BUILDER("MatMulExt").SetBody(BODYFUNC(ib) {
  NodePtr input = ib->GetInput(kIndex0);
  NodePtr other = ib->GetInput(kIndex1);
  auto input_rank = input->shape().size();
  auto other_rank = other->shape().size();
  if (input_rank == 2 && other_rank == 2) {
    auto ret = ib->MatMul(input, other);
    return {ret};
  }
  const ShapeVector &shape1_orig = input->shape();
  const ShapeVector &shape2_orig = other->shape();
  bool is_empty_tensor =
    std::any_of(shape1_orig.begin(), shape1_orig.end(), [](const auto &element) { return element == 0; });
  if (is_empty_tensor) {
    return {ib->Tensor(0, input->dtype())};
  }
  bool transpose_b = other_rank == 1;
  ShapeVector shape_backbone = ops::CheckMatMulShapes(shape1_orig, shape2_orig);
  ShapeVector shape_out = ops::InferShapeRem(shape_backbone, shape1_orig, shape2_orig, transpose_b);
  input = Expand(ib, input, 2);
  other = Expand(ib, other, 2);
  NodePtr ret;
  if (Rank(other) == 2) {
    if (Rank(input) > 2) {
      int64_t new_shape_dim0 = 1;
      for (size_t i = 0; i < shape1_orig.size() - 1; ++i) {
        new_shape_dim0 *= shape1_orig[i];
      }
      std::vector<int64_t> new_shape_vector = {new_shape_dim0, shape1_orig.back()};
      input = ib->Reshape(input, ib->Value(new_shape_vector));
    }
    ret = ib->MatMul(input, other, false, transpose_b);
  } else {
    size_t ndim_aligned = std::max(input_rank, other_rank);
    input = Expand(ib, input, ndim_aligned);
    other = Expand(ib, other, ndim_aligned);
    ShapeVector shape1_aligned = input->shape();
    ShapeVector shape2_aligned = other->shape();
    ShapeVector shape_cur1(shape1_aligned.begin(), shape1_aligned.end() - 2);
    ShapeVector shape_cur2(shape2_aligned.begin(), shape2_aligned.end() - 2);
    const ShapeVector &broadcast_shape1 = ops::GetMatMulExtBroadcastShape(shape_backbone, shape1_orig);
    const ShapeVector &broadcast_shape2 = ops::GetMatMulExtBroadcastShape(shape_backbone, shape2_orig);
    if (input->shape() != broadcast_shape1) {
      input = ib->Emit("BroadcastTo", {input, ib->Value(broadcast_shape1)});
    }
    if (other->shape() != broadcast_shape2) {
      other = ib->Emit("BroadcastTo", {other, ib->Value(broadcast_shape2)});
    }
    input = ib->Reshape(input, ReduceTo3D(input->shape()));
    other = ib->Reshape(other, ReduceTo3D(other->shape()));
    ret = ib->BatchMatMul(input, other, false, transpose_b);
  }
  ret = ib->Reshape(ret, ib->Value(shape_out));
  return {ret};
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
    auto dtype_opt = ops::GetScalarValue<int64_t>(dtype->BuildValue());
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
    auto dtype_opt = ops::GetScalarValue<int64_t>(dtype->BuildValue());
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
    auto dtype_opt = ops::GetScalarValue<int64_t>(dtype->BuildValue());
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
  auto mode_opt = mindspore::ops::GetScalarValue<int64_t>(mode_value_ptr);

  if (mode_opt.value() == ops::RoundingMode::FLOOR) {
    return {ib->Emit("FloorDiv", {input_x, input_y})};
  } else if (mode_opt.value() == ops::RoundingMode::TRUNC) {
    auto div_out = ib->Cast(ib->Div(input_x, input_y), ib->GetDtype(input_x)->type_id());
    return {ib->Emit("Trunc", {div_out})};
  } else {
    MS_LOG(EXCEPTION) << "DivMod abstract failed.";
  }
});

REG_FALLBACK_BUILDER("EqualCount").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(kIndex0);
  const auto &input_y = ib->GetInput(kIndex1);
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
}  // namespace expander
}  // namespace mindspore
