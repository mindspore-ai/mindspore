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

ShapeVector To3D(ShapeVector shape) {
  ShapeVector ret;

  int64_t dim0 = 1;
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    dim0 *= shape[i];
  }
  ret.push_back(dim0);
  ret.push_back(shape[shape.size() - 2]);
  ret.push_back(shape[shape.size() - 1]);
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
    auto ret = ib->BatchMatMul(input, other);
    ret = ib->Reshape(ret, shapes[5]);
    return {ret};
  } else {
    auto input_rank = input->shape().size();
    auto other_rank = other->shape().size();
    if (input_rank == 2 && other_rank == 2) {
      auto ret = ib->MatMul(input, other);
      return {ret};
    }
    const ShapeVector &shape1_orig = input->shape();
    const ShapeVector &shape2_orig = other->shape();
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
      input = ib->Reshape(input, To3D(input->shape()));
      other = ib->Reshape(other, To3D(other->shape()));
      ret = ib->BatchMatMul(input, other, false, transpose_b);
    }
    ret = ib->Reshape(ret, ib->Value(shape_out));
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
}  // namespace expander
}  // namespace mindspore
