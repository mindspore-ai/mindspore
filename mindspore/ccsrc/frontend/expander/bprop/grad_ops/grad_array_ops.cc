/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include "ops/op_utils.h"
#include "ops/array_op_name.h"
#include "ops/array_ops.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "ir/functor.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
namespace {
/**
 * @brief Calculate the shape for gradient of tile to reducing. Cases:
 *        1. dims:    [2, 3], input_shape:    [4, 5] ==>       [2, 4, 3, 5].
 *        2. dims:    [2, 3], input_shape: [4, 5, 6] ==> [1, 4, 2, 5, 3, 6].
 *        3. dims: [2, 3, 4], input_shape:       [5] ==> [2, 1, 3, 1, 4, 5].
 *
 * @param dims Dims argument for op Tile.
 * @param input_shape Shape of input tensor for op Tile.
 * @return std::vector<int64_t> Return shape.
 */
std::vector<int64_t> TileShape(const std::vector<int64_t> &dims, const std::vector<int64_t> &input_shape) {
  int64_t len_multi = static_cast<int64_t>(dims.size());
  int64_t len_shape = static_cast<int64_t>(input_shape.size());
  int64_t len_cmp = len_multi - len_shape;
  auto max_len = std::max(len_multi, len_shape);
  int64_t i = 0;
  int64_t j = 0;
  std::vector<int64_t> res;
  auto res_sz = static_cast<size_t>(2 * max_len);
  res.reserve(res_sz);
  while (i < max_len && j < max_len) {
    auto idx_i = LongToSize(i);
    auto idx_j = LongToSize(j);
    if (len_cmp == 0) {
      res.push_back(dims[idx_i]);
      res.push_back(input_shape[idx_j]);
      i++;
      j++;
    } else if (len_cmp > 0) {
      res.push_back(dims[idx_i]);
      res.push_back(1);
      i++;
      len_cmp--;
    } else {
      res.push_back(1);
      res.push_back(input_shape[idx_j]);
      j++;
      len_cmp++;
    }
  }

  return res;
}

inline NodePtr MinOrMaxOpGetMask(BpropBuilder *ib, const NodePtr &x, const NodePtr &out) {
  auto out_is_nan = ib->IsNanFunc(out);
  auto input_is_nan = [&x](Emitter *e) -> NodePtrList { return {e->IsNanFunc(x)}; };
  auto input_equal_out = [&x, &out](Emitter *e) -> NodePtrList { return {e->Equal(x, out)}; };
  return ib->Conditional(out_is_nan, input_is_nan, input_equal_out);
}

NodePtr MinOrMaxOpGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &out, const NodePtr &dout) {
  auto mask = MinOrMaxOpGetMask(ib, x, out);
  auto x_zeros = ib->Zeros(x);
  auto mask_sum = ib->Emit("SumExt", {mask, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone)});
  auto grad_div_mask_sum = ib->Div(dout, ib->Cast(mask_sum, ib->GetDtype(dout)));
  auto dx = ib->Emit("MaskedFill", {x_zeros, mask, grad_div_mask_sum});
  return {dx};
}
}  // namespace

DEF_PURE_SHAPE_CALC(g_gather_drop_negative)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto is_pos = inputs.at(1);
    auto gather_rank = inputs.at(0).size();
    auto is_pos_rank = is_pos.size();
    std::vector<int64_t> res_shape(is_pos.begin(), is_pos.end());
    if (gather_rank > is_pos_rank) {
      auto expand_len = gather_rank - is_pos_rank;
      for (size_t i = 0; i < expand_len; ++i) {
        res_shape.push_back(1);
      }
    }
    return {res_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto gather = inputs.at(0);
    auto is_pos = inputs.at(1);
    if (!unknown_inputs.empty() || IsDynamicRank(gather) || IsDynamicRank(is_pos)) {
      return {-1};
    }
    auto gather_rank = gather.size();
    auto is_pos_rank = is_pos.size();
    return {SizeToLong(std::max(gather_rank, is_pos_rank))};
  });
NodePtrList GatherDropNegatives(BpropBuilder *ib, const NodePtr &params, const NodePtr &ids,
                                const NodePtr &zero_clipped_indices_param = nullptr,
                                const NodePtr &is_positive_param = nullptr) {
  NodePtr zero_clipped_indices = zero_clipped_indices_param;
  if (zero_clipped_indices_param == nullptr) {
    zero_clipped_indices = ib->Maximum(ids, ib->ZerosLike(ids));
  }
  auto gathered = ib->Gather(params, zero_clipped_indices, 0);

  NodePtr is_positive = is_positive_param;
  if (is_positive_param == nullptr) {
    is_positive = ib->GreaterEqual(ids, ib->Tensor(0, ib->GetDtype(ids)));
    auto broadcastable_shape = ib->GetShape(is_positive);
    auto gathered_shape = ib->GetShape(gathered);
    if (IsDynamic(broadcastable_shape) || IsDynamic(gathered_shape)) {
      auto is_positive_shape = ib->ShapeCalc(g_gather_drop_negative, {gathered, is_positive})[0];
      is_positive = ib->Reshape(is_positive, is_positive_shape);
      auto shape_gather = ib->Shape(gathered, true);
      is_positive = ib->LogicalAnd(is_positive, ib->Fill(1.0, shape_gather, TypeId::kNumberTypeBool));
    } else {
      auto back_size = ib->GetShape(gathered).size() - ib->GetShape(is_positive).size();
      for (size_t i = 0; i < back_size; ++i) {
        broadcastable_shape.push_back(1);
      }
      is_positive = ib->Reshape(is_positive, broadcastable_shape);
      auto ones = ib->Fill(1.0, gathered_shape, TypeId::kNumberTypeBool);
      is_positive = ib->LogicalAnd(is_positive, ones);
    }
  }
  auto zero_slice = ib->ZerosLike(gathered);
  return {ib->Select(is_positive, gathered, zero_slice), zero_clipped_indices, is_positive};
}

NodePtrList UnsortedSegmentMinOrMaxGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &segment_ids,
                                        const NodePtr &num_segments, const NodePtr &out, const NodePtr &dout) {
  auto temp_outs = GatherDropNegatives(ib, out, segment_ids, nullptr, nullptr);
  constexpr size_t out_size = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() == out_size, "Outputs' size should be 3.");
  auto gathered_outputs = temp_outs[0];
  auto zero_clipped_indices = temp_outs[1];
  auto is_positive = temp_outs[2];

  auto tmp = ib->Equal(x, gathered_outputs);
  auto is_selected = ib->LogicalAnd(tmp, is_positive);
  auto num_selected = ib->UnsortedSegmentSum(ib->Cast(is_selected, ib->GetDtype(dout)), segment_ids, num_segments);
  auto weighted_grads = ib->RealDiv(dout, num_selected);
  auto temp_outs_2 = GatherDropNegatives(ib, weighted_grads, nullptr, zero_clipped_indices, is_positive);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grads = temp_outs_2[0];
  auto zeros = ib->ZerosLike(gathered_grads);
  return {ib->Select(is_selected, gathered_grads, zeros), ib->OutZeros(segment_ids), ib->OutZeros(num_segments)};
}

NodePtrList SegmentMinOrMaxGrad(BpropBuilder *ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto output = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    output = ib->Cast(output, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto zero_value = ib->Value<int64_t>(0);
  auto gathered_outputs = ib->Gather(output, segment_ids, zero_value);
  auto is_selected = ib->Equal(input_x, gathered_outputs);
  const int64_t max_len = 1000000;
  auto num_selected =
    ib->Emit("SegmentSum", {ib->Cast(is_selected, kFloat32), segment_ids}, {{"max_length", MakeValue(max_len)}});
  auto weighted_grads = ib->Cast(ib->Div(dout, num_selected), ib->GetDtype(dout));
  auto gathered_grads = ib->Gather(weighted_grads, segment_ids, zero_value);
  auto dx = ib->Select(is_selected, gathered_grads, ib->ZerosLike(input_x));
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->OutZeros(segment_ids)};
}

NodePtrList TensorScatterPossibleReplacement(BpropBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto x_indicators = ib->Cast(ib->Equal(x, out), kInt32);
  auto possibly_updated = ib->GatherNd(out, indices);
  auto out_indicators = ib->Cast(ib->Equal(updates, possibly_updated), kInt32);
  auto input_shape = ib->Shape(x);
  auto scattered_out_indicators = ib->ScatterNd(indices, out_indicators, input_shape);
  auto indicators = ib->Add(x_indicators, scattered_out_indicators);
  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    dx = ib->RealDiv((ib->Mul(dout, (ib->Cast(x_indicators, ib->GetDtype(dout))))),
                     (ib->Cast(indicators, ib->GetDtype(dout))));
    dx = ib->Cast(dx, ib->GetDtype(x));
  } else {
    dx = ib->OutZeros(x);
  }
  NodePtr update_grad = nullptr;
  if (updates->need_compute_grad_out()) {
    update_grad = ib->Mul((ib->GatherNd(ib->RealDiv(dout, (ib->Cast(indicators, ib->GetDtype(dout)))), indices)),
                          (ib->Cast(out_indicators, ib->GetDtype(dout))));
    update_grad = ib->Cast(update_grad, ib->GetDtype(updates));
  } else {
    update_grad = ib->OutZeros(updates);
  }

  return {dx, ib->OutZeros(indices), update_grad};
}

ShapeArray RegenerateOutputShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x_shape = inputs.at(kIndex0);
  auto indices_shape = inputs.at(kIndex1);
  auto axis = inputs.at(kIndex2);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(kIndex3);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[kIndex0];

  auto out_shape = RegenerateOutputShape(x_shape, indices_shape, axis_value, batch_dims_value);
  return {out_shape};
}

std::vector<int64_t> RegenerateOutputInferFunc(const ShapeArray &inputs, const HashSet<size_t> &invalid_indices) {
  constexpr size_t inputs_num = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x = inputs.at(kIndex0);
  auto indices = inputs.at(kIndex1);
  auto batch_dims = inputs.at(kIndex3);
  if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(indices) || IsDynamicRank(batch_dims)) {
    return {-1};
  }

  auto x_rank = x.size();
  auto indices_rank = indices.size();
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  return {SizeToLong(x_rank + indices_rank - LongToSize(batch_dims_value))};
}

ShapeArray PermsShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x_shape = inputs.at(kIndex0);
  auto dout_shape = inputs.at(kIndex1);
  auto indices_shape = inputs.at(kIndex2);
  auto axis = inputs.at(kIndex3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(kIndex4);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  auto perm_1 = GenerateShapeIndex(dout_shape, indices_shape, axis_value, batch_dims_value);
  auto perm_2 = GenerateInverseIndex(x_shape, axis_value, batch_dims_value);

  return {perm_1, perm_2};
}

std::vector<int64_t> PermsInferFunc(const ShapeArray &inputs, const HashSet<size_t> &invalid_indices) {
  auto x = inputs.at(kIndex0);
  auto dout = inputs.at(kIndex1);
  if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(dout)) {
    return {-1, -1};
  }

  return {SizeToLong(dout.size()), SizeToLong(x.size())};
}

DEF_PURE_SHAPE_CALC(g_calc_num_segment)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shp = inputs.at(kIndex0);
    auto axis_v = inputs.at(kIndex1)[0];
    axis_v = NormalizeAxis(axis_v, x_shp.size());
    return {{x_shp[LongToSize(axis_v)]}};
  })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });
NodePtr CalcNumSegment(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis) {
  MS_EXCEPTION_IF_NULL(ib);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(axis);
  auto num_segment = ib->ShapeCalc(g_calc_num_segment, {x, axis}, {1})[0];
  if (num_segment->input_type() == InputType::kConstant) {
    auto num_segment_value = GetIntList(num_segment);
    MS_EXCEPTION_IF_CHECK_FAIL(num_segment_value.size() == 1,
                               "The num_segment should be a int for gradient of Gather.");
    num_segment = ib->Value(num_segment_value[0]);
  } else {
    num_segment = ib->TupleGetItem(num_segment, 0);
  }
  return num_segment;
}

ShapeArray GatherReshapeShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 5.");

  auto values_shape = inputs.at(0);
  auto indices_shape = inputs.at(1);
  auto x_shape = inputs.at(2);
  auto axis = inputs.at(3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  if (axis_value < 0) {
    axis_value += SizeToLong(x_shape.size());
  }

  auto batch_dims = inputs.at(4);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  MS_EXCEPTION_IF_CHECK_FAIL(x_shape.size() > LongToSize(axis_value), "axis should within interval: [0, params_rank).");
  MS_EXCEPTION_IF_CHECK_FAIL(axis_value >= batch_dims_value, "axis can not less than batch_dims.");
  int64_t batch_size = 1;
  for (int64_t i = 0; i < batch_dims_value; i++) {
    batch_size *= x_shape[i];
  }
  auto axis_dim = x_shape[axis_value];

  std::vector<int64_t> values_reshape = {-1};
  (void)values_reshape.insert(values_reshape.end(), values_shape.begin() + batch_dims_value, values_shape.end());

  std::vector<int64_t> indices_reshape = {-1};
  (void)indices_reshape.insert(indices_reshape.end(), indices_shape.begin() + batch_dims_value, indices_shape.end());

  std::vector<int64_t> delta_reshape = {batch_size};
  auto indices_rank = SizeToLong(indices_reshape.size());
  for (int64_t i = 0; i < indices_rank - 1; i++) {
    delta_reshape.push_back(1);
  }

  std::vector<int64_t> params_grad_reshape(values_shape.begin(), values_shape.begin() + batch_dims_value);
  params_grad_reshape.push_back(axis_dim);
  (void)params_grad_reshape.insert(params_grad_reshape.end(), values_reshape.begin() + indices_rank,
                                   values_reshape.end());

  ShapeArray res = {values_reshape, indices_reshape, delta_reshape, params_grad_reshape};
  return res;
}

std::vector<int64_t> GatherReshapeInferFunc(const ShapeArray &inputs, const HashSet<size_t> &invalid_indices) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 5.");

  auto values = inputs.at(0);
  auto indices = inputs.at(1);
  auto params_grad = inputs.at(2);
  auto batch_dims = inputs.at(4);

  constexpr size_t return_num = 4;
  if (!invalid_indices.empty() || IsDynamicRank(values) || IsDynamicRank(indices) || IsDynamicRank(params_grad) ||
      IsDynamicRank(batch_dims)) {
    return ShapeVector(return_num, -1);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  auto values_rank = SizeToLong(values.size()) - batch_dims_value + 1;
  auto indices_rank = SizeToLong(indices.size()) - batch_dims_value + 1;
  auto delta_rank = indices_rank;
  auto params_grad_rank = SizeToLong(params_grad.size());

  ShapeVector res = {values_rank, indices_rank, delta_rank, params_grad_rank};
  return res;
}

class CalBatchGatherShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_CalBatchGather", CalBatchGatherShapeCalc)
  CalBatchGatherShapeCalc(int64_t axis, int64_t batch_dims)
      : ShapeCalcFunctor("ShapeCalc_CalBatchGather"), axis_(axis), batch_dims_(batch_dims) {}
  ValuePtr ToValue() const override {
    auto values = {MakeValue(axis_), MakeValue(batch_dims_)};
    return std::make_shared<ValueTuple>(values);
  }
  void FromValue(const ValuePtr &value) override {
    auto values = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(values);
    if (values->value().size() != i2) {
      MS_LOG(EXCEPTION) << "CalBatchGatherShapeCalc's value size should be 2, but got " << values->value().size();
    }
    axis_ = GetValue<int64_t>(values->value()[0]);
    batch_dims_ = GetValue<int64_t>(values->value()[1]);
  }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shp = inputs.at(0);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < batch_dims_; i++) {
      batch_size *= x_shp[i];
    }
    ShapeVector aixs_dim = {x_shp[axis_]};
    ShapeVector limit = {batch_size * x_shp[axis_]};
    return {aixs_dim, limit};
  }
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override { return {1, 1}; }

 protected:
  int64_t axis_{0};
  int64_t batch_dims_{0};
};
REG_FUNCTOR("ShapeCalc_CalBatchGather", CalBatchGatherShapeCalc);
DEF_PURE_SHAPE_CALC(g_gather_reshape).SetCalc(GatherReshapeShapeFunc).SetInfer(GatherReshapeInferFunc);

class ResizeNearestNeighborV2ShapeCalc : public ShapeCalcFunctor {
 public:
  explicit ResizeNearestNeighborV2ShapeCalc(bool is_nchw)
      : ShapeCalcFunctor("ShapeCalc_ResizeNearestNeighborV2"), is_nchw_(is_nchw) {}
  DECLARE_SHAPE_CALC("ShapeCalc_ResizeNearestNeighborV2", ResizeNearestNeighborV2ShapeCalc)
  ValuePtr ToValue() const override { return MakeValue(is_nchw_); }
  void FromValue(const ValuePtr &value) override { is_nchw_ = GetValue<bool>(value); }

  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs.at(0);
    ShapeVector grad_in_size = (is_nchw_ ? GetShapeByRange(x_shape, 2, 4) : GetShapeByRange(x_shape, 1, 3));
    if (grad_in_size.size() != i2) {
      MS_LOG(EXCEPTION) << "For ResizeNearestNeighborV2Grad, size's rank should be 2, but got " << grad_in_size.size();
    }
    return {grad_in_size};
  }
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override { return {2}; }

 protected:
  bool is_nchw_{true};
};
REG_FUNCTOR("ShapeCalc_ResizeNearestNeighborV2", ResizeNearestNeighborV2ShapeCalc);

NodePtr CalBatchGather(BpropBuilder *ib, const NodePtr &values, const NodePtr &indices, const NodePtr &x, int64_t axis,
                       int64_t batch_dims) {
  auto reshape_shape =
    ib->ShapeCalc(g_gather_reshape, {values, indices, x, ib->Tensor(axis), ib->Tensor(batch_dims)}, {3, 4});
  constexpr size_t reshape_size = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_shape.size() == reshape_size, "reshape_shape should equal to 4.");
  auto values_reshape = reshape_shape[0];
  auto indices_reshape = reshape_shape[1];
  auto delta_reshape = reshape_shape[2];
  auto params_grad_reshape = reshape_shape[3];

  auto values_rshp = ib->Reshape(values, values_reshape);
  auto indices_rshp = ib->Reshape(indices, indices_reshape);
  auto res = ib->ShapeCalc(std::make_shared<CalBatchGatherShapeCalc>(axis, batch_dims), {x});
  auto axis_dim = res[0];
  auto limit = res[1];
  auto delta = ib->Range(ib->Value<int64_t>(0), ib->TupleGetItem(limit, 0), ib->TupleGetItem(axis_dim, 0));
  delta = ib->Reshape(delta, delta_reshape);
  indices_rshp = ib->Add(indices_rshp, delta);
  auto params_grad = ib->UnsortedSegmentSum(values_rshp, indices_rshp, ib->TupleGetItem(limit, 0));
  params_grad = ib->Reshape(params_grad, params_grad_reshape);
  return params_grad;
}

bool IsMutable(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value_ptr = node->BuildValue();
  if (value_ptr != nullptr &&
      (value_ptr->isa<ValueSequence>() || value_ptr->isa<Scalar>() || value_ptr->isa<tensor::Tensor>())) {
    return false;
  }
  return true;
}

DEF_PURE_SHAPE_CALC(g_regenerate_output).SetCalc(RegenerateOutputShapeFunc).SetInfer(RegenerateOutputInferFunc);
DEF_PURE_SHAPE_CALC(g_perms).SetCalc(PermsShapeFunc).SetInfer(PermsInferFunc);
NodePtrList BinopGather(BpropBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto ori_indices = indices;  // indices may be changed latter.
  auto axis = ib->GetInput(kIndex2);
  auto batch_dims_ptr = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shp = ib->GetShape(x);
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);

  if (out_shp.empty()) {
    dout = ib->ExpandDims(dout, -1);
  }
  int64_t axis_v = CheckRange(GetIntValue(axis), SizeToLong(x_shp.size()));
  auto batch_dims = GetIntValue(batch_dims_ptr);
  auto ind_shp_size = SizeToLong(ind_shp.size());
  while (batch_dims < 0) {
    batch_dims += ind_shp_size;
  }

  auto is_axis_mutable = IsMutable(axis);
  if ((!is_axis_mutable && (IsDynamicRank(x_shp) || IsDynamicRank(ind_shp) || IsDynamicRank(out_shp))) ||
      (is_axis_mutable && (IsDynamic(x_shp) || IsDynamic(ind_shp) || IsDynamic(out_shp)))) {
    auto batch_dims_tensor = ib->Tensor(batch_dims, kInt64);
    if (ind_shp.empty()) {
      indices = ib->ExpandDims(indices, -1);
      auto out_shp1 = ib->ShapeCalc(g_regenerate_output, {x, indices, axis, batch_dims_tensor}, {kIndex2, kIndex3})[0];
      dout = ib->Reshape(dout, out_shp1);
    }
    // Calculate perm.
    auto perms = ib->ShapeCalc(g_perms, {x, dout, indices, axis, batch_dims_tensor}, {kIndex3, kIndex4});
    const size_t perm_num = 2;
    MS_EXCEPTION_IF_CHECK_FAIL(perms.size() == perm_num, "Perms number should be 2 for gradient of Gather.");
    auto perm_1 = perms[0];
    auto perm_2 = perms[1];
    auto values_transpose = ib->Transpose(dout, perm_1);
    NodePtr x_grad = nullptr;
    if (batch_dims > 0) {
      x_grad = CalBatchGather(ib, values_transpose, indices, x, axis_v, batch_dims);
    } else {
      auto num_segment = CalcNumSegment(ib, x, axis);
      x_grad = ib->UnsortedSegmentSum(values_transpose, indices, num_segment);
    }
    x_grad = ib->Transpose(x_grad, perm_2);
    return {x_grad, ib->OutZeros(ori_indices), ib->OutZeros(axis), ib->OutZeros(batch_dims_ptr)};
  }

  if (ind_shp.empty()) {
    indices = ib->ExpandDims(indices, -1);
    ind_shp = ib->GetShape(indices);
    auto out_shp1 = RegenerateOutputShape(x_shp, ind_shp, axis_v);
    dout = ib->Reshape(dout, out_shp1);
  }

  out_shp = ib->GetShape(dout);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_v, batch_dims);
  auto values_transpose = ib->Transpose(dout, perm_1);
  NodePtr x_grad = nullptr;
  if (batch_dims > 0) {
    x_grad = CalBatchGather(ib, values_transpose, indices, x, axis_v, batch_dims);
  } else {
    auto num_segment = CalcNumSegment(ib, x, axis);
    x_grad = ib->UnsortedSegmentSum(values_transpose, indices, num_segment);
  }
  auto perm_2 = GenerateInverseIndex(x_shp, axis_v, batch_dims);
  auto params_grad = ib->Transpose(x_grad, perm_2);
  return {params_grad, ib->OutZeros(ori_indices), ib->OutZeros(axis), ib->OutZeros(batch_dims_ptr)};
}

ShapeArray ConcatOffsetCal(const ShapeArray &input_shapes, size_t axis_s) {
  ShapeArray res;
  auto rank = input_shapes[0].size();
  auto input_num = input_shapes.size();
  int64_t sum_axis = 0;
  for (size_t i = 0; i < input_num; ++i) {
    std::vector<int64_t> offset(rank, 0);
    offset[axis_s] = sum_axis;
    sum_axis += input_shapes.at(i)[axis_s];
    res.push_back(offset);
  }
  return res;
}

NodePtrList ConcatBpropStatic(BpropBuilder *ib, const NodePtr &dout, const ShapeArray &input_shapes, int64_t axis) {
  auto rank = input_shapes[0].size();
  auto axis_s = LongToSize(NormalizeAxis(axis, rank));

  bool is_uniform = true;
  auto input_nums = input_shapes.size();
  for (size_t i = 0; i < input_nums; ++i) {
    if (input_shapes[i].size() != rank) {
      MS_EXCEPTION(ValueError) << "For gradient of 'Concat', input shapes [" << i
                               << "] and input shapes [0] must have same rank, but got: " << input_shapes[i].size()
                               << " vs " << rank;
    }
    if (input_shapes[i][axis_s] != input_shapes[0][axis_s]) {
      is_uniform = false;
    }
  }

  if (is_uniform) {
    auto long_nums = SizeToLong(input_nums);
    auto dx = ib->Emit(kSplitOpName, {dout, ib->EmitValue(MakeValue(axis)), ib->EmitValue(MakeValue(long_nums))},
                       {{"num_split", MakeValue(long_nums)}});
    return {dx};
  }

  NodePtrList res;
  auto offsets = ConcatOffsetCal(input_shapes, axis_s);
  for (size_t i = 0; i < input_nums; ++i) {
    auto offset_value = ib->Value(offsets[i]);
    auto slice_out = ib->Emit(kSliceOpName, {dout, offset_value, ib->Value(input_shapes[i])});
    res.push_back(slice_out);
  }
  return {ib->MakeTuple(res)};
}

NodePtrList StackBpropFunc(BpropBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto num = ib->GetAttr("num");
  auto ret = ib->Emit("Unstack", {dout}, {{"num", num}, {"axis", ib->GetAttr("axis")}});

  auto x_abs = x->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  bool is_list = x_abs->isa<abstract::AbstractList>();
  if (is_list) {
    NodePtrList res;
    auto num_v = LongToSize(GetValue<int64_t>(num));
    for (size_t i = 0; i < num_v; ++i) {
      res.push_back(ib->TupleGetItem(ret, i));
    }
    return {ib->MakeList(res)};
  }
  return {ret};
}

NodePtrList BinopGatherDGradCommon(BpropBuilder *ib, const std::string &op_name) {
  auto dim = LongToSize(GetValue<int64_t>(ib->GetAttr("dim")));
  auto index = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  ShapeVector x_shp;
  if (op_name == "GatherDGrad") {
    x_shp = GetValue<ShapeVector>(ib->GetAttr("shape"));
  } else {
    x_shp = ib->GetShape(x);
  }
  auto index_shp = ib->GetShape(index);
  int64_t dim_before_axis = 1;
  for (size_t i = 0; i < dim; ++i) {
    dim_before_axis *= x_shp[i];
  }
  auto dim_at_axis_index = index_shp[dim];
  auto dim_at_axis_output = x_shp[dim];
  int64_t dim_after_axis = 1;
  for (size_t i = dim + 1; i < x_shp.size(); ++i) {
    dim_after_axis *= x_shp[i];
  }
  auto element = (dim_before_axis * dim_at_axis_index) * dim_after_axis;
  auto index_type = ib->GetDtype(index);
  auto id = ib->Tensor(Range(element), index_type);
  auto i = ib->FloorDiv(id, ib->Tensor((dim_at_axis_index * dim_after_axis), index_type));
  auto k = ib->FloorMod(id, ib->Tensor(dim_after_axis, index_type));
  auto less = ib->Less(index, ib->Tensor(0, index_type));
  auto j = ib->Cast(less, index_type);
  auto j_read = ib->Add((ib->Mul(ib->Tensor(dim_at_axis_index, index_type), j)), index);
  auto j_read_reshape = ib->Reshape(j_read, {-1});
  auto i_after = ib->Mul(i, ib->Tensor(dim_at_axis_output * dim_after_axis, index_type));
  auto read_id = ib->Add((ib->Add(i_after, (ib->Mul(j_read_reshape, ib->Tensor(dim_after_axis, index_type))))), k);
  auto dout_reshape = ib->Reshape(dout, {-1});
  auto dx = ib->Gather(dout_reshape, read_id, ib->Tensor(0));
  dx = ib->Reshape(dx, ib->GetShape(x));
  return {ib->OutZeros(index), dx};
}

class SortShapeCalc1 : public ShapeCalcFunctor {
 public:
  DECLARE_SHAPE_CALC("ShapeCalc_Sort_1", SortShapeCalc1)
  explicit SortShapeCalc1(int64_t axis) : ShapeCalcFunctor("ShapeCalc_Sort_1"), axis_(axis) {}
  ValuePtr ToValue() const override { return MakeValue(axis_); }
  void FromValue(const ValuePtr &value) override { axis_ = GetValue<int64_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs[0];
    auto x_rank = x_shape.size();
    auto recorrect_axis = NormalizeAxis(axis_, x_rank);
    ShapeVector transposition;
    ShapeVector invert_perm;
    if (LongToSize(recorrect_axis + 1) == x_rank) {
      // A (0, 1, 2, ...) will change Transpose as a copy-like operator.
      // This can delete two control flow block.
      transposition = Range(SizeToLong(x_rank));
      invert_perm = Range(SizeToLong(x_rank));
    } else {
      transposition = GetTransposition(recorrect_axis, SizeToLong(x_rank));
      invert_perm = InvertPermutation(transposition);
    }

    auto k = x_shape.at(LongToSize(recorrect_axis));
    return {{k}, transposition, invert_perm};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) const override {
    auto x = inputs.at(0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {1, -1, -1};
    }

    auto x_rank = SizeToLong(x.size());
    return {1, x_rank, x_rank};
  }

 protected:
  int64_t axis_{0};
};
REG_FUNCTOR("ShapeCalc_Sort_1", SortShapeCalc1);

std::pair<std::vector<bool>, std::vector<std::vector<int64_t>>> DynBroadcastGradientArgsSelect(
  const std::vector<int64_t> &cond_shape, const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  auto cond_size = cond_shape.size();
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  ShapeVector shape[kDim3] = {cond_shape, x_shape, y_shape};
  auto n = std::max(cond_size, std::max(x_size, y_size));
  std::vector<bool> need_shapecalc = {false, false, false};
  std::vector<std::vector<int64_t>> reduce_axis(kDim3);
  if (IsDynamicRank(shape[kIndex0]) || IsDynamicRank(shape[kIndex1]) || IsDynamicRank(shape[kIndex2])) {
    return {{true, true, true}, reduce_axis};
  }
  for (size_t i = n; i >= 1; i--) {
    int64_t dim_value[kDim3] = {cond_size < i ? 1 : shape[kIndex0][cond_size - i],
                                x_size < i ? 1 : shape[kIndex1][x_size - i],
                                y_size < i ? 1 : shape[kIndex2][y_size - i]};
    const int64_t reduce_idx = SizeToLong(n - i);
    bool is_dynamic = false;
    if (dim_value[kIndex1] == dim_value[kIndex0] && dim_value[kIndex2] == dim_value[kIndex0]) {
      if (dim_value[kIndex0] == -1) {
        need_shapecalc[kIndex0] = need_shapecalc[kIndex1] = need_shapecalc[kIndex2] = true;
        break;
      }
    } else {
      for (size_t j = 0; j < kDim3; j++) {
        if (dim_value[j] == 1) {
          (void)reduce_axis[j].emplace_back(reduce_idx);
        } else if (dim_value[j] > 0) {
          is_dynamic = true;
        }
      }
      for (size_t j = 0; j < kDim3; j++) {
        if (is_dynamic && dim_value[j] == -1) {
          need_shapecalc[j] = true;
          (void)reduce_axis[j].emplace_back(reduce_idx);
        }
      }
    }
  }
  return {need_shapecalc, reduce_axis};
}

ShapeArray BroadcastGradientArgsInferValueSelect(const ShapeVector &cond_shape, const ShapeVector &x_shape,
                                                 const ShapeVector &y_shape) {
  ShapeArray bc_axis;
  if (cond_shape == x_shape && cond_shape == y_shape) {
    (void)bc_axis.emplace_back(ShapeVector{});
    (void)bc_axis.emplace_back(ShapeVector{});
    (void)bc_axis.emplace_back(ShapeVector{});
    return bc_axis;
  }
  ShapeVector grad_cond_reduce_idx;
  ShapeVector grad_x_reduce_idx;
  ShapeVector grad_y_reduce_idy;
  auto cond_size = cond_shape.size();
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  auto n = std::max(cond_size, std::max(x_size, y_size));
  for (size_t i = n; i >= 1; i--) {
    auto cond_i = cond_size < i ? 1 : cond_shape[cond_size - i];
    auto x_i = x_size < i ? 1 : x_shape[x_size - i];
    auto y_i = y_size < i ? 1 : y_shape[y_size - i];
    const int64_t reduce_idx = SizeToLong(n - i);
    if (cond_i == x_i && cond_i == y_i) {
      continue;
    }
    if (cond_i == 1) {
      grad_cond_reduce_idx.push_back(reduce_idx);
    }
    if (x_i == 1) {
      grad_x_reduce_idx.push_back(reduce_idx);
    }
    if (y_i == 1) {
      grad_y_reduce_idy.push_back(reduce_idx);
    }
  }

  (void)bc_axis.emplace_back(std::move(grad_cond_reduce_idx));
  (void)bc_axis.emplace_back(std::move(grad_x_reduce_idx));
  (void)bc_axis.emplace_back(std::move(grad_y_reduce_idy));
  return bc_axis;
}

DEF_PURE_SHAPE_CALC(g_select_broadcast)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto shape_cond = inputs.at(kIndex0);
    auto shape_x = inputs.at(kIndex1);
    auto shape_y = inputs.at(kIndex2);
    return BroadcastGradientArgsInferValueSelect(shape_cond, shape_x, shape_y);
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    constexpr int64_t kShapeDimAny = -1;
    return {kShapeDimAny, kShapeDimAny, kShapeDimAny};
  });

NodePtrList DynBinopGradSelect(BpropBuilder *ib, const NodePtr &cond, const NodePtr &x, const NodePtr &y,
                               const NodePtr &dout, const NodePtr &dx, const NodePtr &dy, size_t shift = 0UL) {
  NodePtr inputs[] = {cond, x, y};
  NodePtrList reduce = {dout, dx, dy};
  ShapeVector shape[] = {ib->GetShape(inputs[kIndex0]), ib->GetShape(inputs[kIndex1]), ib->GetShape(inputs[kIndex2])};
  auto [need_shapecalc, reduce_axis] = DynBroadcastGradientArgsSelect(shape[kIndex0], shape[kIndex1], shape[kIndex2]);
  NodePtrList broadcast_axes;
  if (need_shapecalc[kIndex0] || need_shapecalc[kIndex1] || need_shapecalc[kIndex2]) {
    broadcast_axes = ib->ShapeCalc(g_select_broadcast, {inputs[kIndex0], inputs[kIndex1], inputs[kIndex2]});
  }
  for (size_t i = 1; i < kDim3; i++) {
    auto dout_shape = ib->GetShape(reduce[i]);
    if (!need_shapecalc[i] && IsDynamicRank(dout_shape)) {
      MS_LOG(WARNING) << "The dynamic shape inference of" << reduce[i]->ToString() << " is overly generalized.";
    }
    if (!need_shapecalc[i] && !IsDynamicRank(dout_shape)) {
      if (!reduce_axis[i].empty()) {
        reduce[i] = ib->SumExt(reduce[i], ib->Value<ShapeVector>(reduce_axis[i]),
                               ib->Value<bool>(dout_shape.size() == shape[i].size()));
      }
      if (ib->GetRank(reduce[i]) != shape[i].size()) {
        reduce[i] = ib->Reshape(reduce[i], ib->Shape(inputs[i]));
      }
    } else {
      bool keep_dims = (!IsDynamicRank(shape[kIndex0]) && !IsDynamicRank(shape[kIndex1]) &&
                        !IsDynamicRank(shape[kIndex2]) && shape[i].size() >= shape[i ^ 1].size());
      reduce[i] = ib->ReduceSum(reduce[i], broadcast_axes[i], keep_dims, true);
      reduce[i] = ib->Reshape(reduce[i], ib->Shape(inputs[i]));
    }
  }
  return reduce;
}

NodePtr StaticBinopGradSelect(BpropBuilder *ib, const NodePtr &dx, const ShapeArray &shape,
                              const ShapeArray &broadcast_shape, size_t shift, size_t index, bool *is_dynamic_shape) {
  NodePtr reduce_dx = dx;
  auto shape_dynamic_dims = std::count_if(shape[index].begin(), shape[index].end(), [](int64_t x) { return x <= -1; });
  if (broadcast_shape[kIndex0].empty() || broadcast_shape[kIndex1].empty() || broadcast_shape[kIndex2].empty()) {
    if (broadcast_shape[index].empty()) {
      if (shift) {
        std::vector<int64_t> axis(broadcast_shape[index ^ 1].size());
        std::iota(axis.begin(), axis.end(), 0LL);
        reduce_dx = ib->SumExt(reduce_dx, ib->Value<ShapeVector>(axis), ib->Value(false));
      } else {
        reduce_dx = ib->SumExt(reduce_dx, ib->EmitValue(kNone), ib->Value(false));
      }
    }
  } else if (!IsDynamic(broadcast_shape[kIndex0]) && !IsDynamic(broadcast_shape[kIndex1]) &&
             !IsDynamic(broadcast_shape[kIndex2]) && shape_dynamic_dims <= 1) {
    std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgsInferValueSelect(
      broadcast_shape[kIndex0], broadcast_shape[kIndex1], broadcast_shape[kIndex2]);
    if (!bc_axis[index].empty()) {
      reduce_dx = ib->SumExt(reduce_dx, ib->Value<ShapeVector>(bc_axis[index]),
                             ib->Value<bool>(ib->GetRank(reduce_dx) == shape[index].size()));
    }
    reduce_dx = ib->Reshape(reduce_dx, shape[index]);
  } else {
    *is_dynamic_shape = true;
  }
  return reduce_dx;
}

NodePtrList BinopGradSelect(BpropBuilder *ib, const NodePtr &cond, const NodePtr &x, const NodePtr &y,
                            const NodePtr &dout, const NodePtr &dx, const NodePtr &dy, size_t shift = 0UL) {
  // Grad definition for where/select operations with shift.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  NodePtrList inputs{cond, x, y};
  ShapeArray shape{ib->GetShape(inputs[kIndex0]), ib->GetShape(inputs[kIndex1]), ib->GetShape(inputs[kIndex2])};
  NodePtrList reduce = {dout, dx, dy};
  if (IsDynamicRank(shape[kIndex0]) || IsDynamicRank(shape[kIndex1]) || IsDynamicRank(shape[kIndex2])) {
    return DynBinopGradSelect(ib, cond, x, y, dout, dx, dy, shift);
  }
  if (shape[kIndex0].size() <= shift && shape[kIndex0].size() == shape[kIndex1].size() &&
      shape[kIndex0].size() == shape[kIndex2].size()) {
    return reduce;
  }
  ShapeArray broadcast_shape(kDim3);
  for (size_t i = 0; i < kDim3; i++) {
    broadcast_shape[i] = ShapeVector(shape[i].begin(), shape[i].end() - shift);
  }
  bool is_x_shape_dynamic = false;
  bool is_y_shape_dynamic = false;
  if (dx != nullptr) {
    reduce[kIndex1] =
      StaticBinopGradSelect(ib, reduce[kIndex1], shape, broadcast_shape, shift, kIndex1, &is_x_shape_dynamic);
  }
  if (dy != nullptr) {
    reduce[kIndex2] =
      StaticBinopGradSelect(ib, reduce[kIndex2], shape, broadcast_shape, shift, kIndex2, &is_y_shape_dynamic);
  }
  if (is_x_shape_dynamic || is_y_shape_dynamic) {
    return DynBinopGradSelect(ib, cond, x, y, dout, dx, dy, shift);
  }
  return reduce;
}

REG_BPROP_BUILDERS_BEGIN(GradArrayOps)
REG_BPROP_BUILDER("GatherD").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("GatherDGradV2", {x, dim, index, dout});
  return {dx, ib->OutZeros(dim), ib->OutZeros(index)};
});

REG_BPROP_BUILDER("GatherDGrad").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGrad");
});

REG_BPROP_BUILDER("GatherDGradV2").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGradV2");
});

REG_BPROP_BUILDER("SparseGatherV2").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto axis_int = CheckRange(GetIntList(axis->BuildValue())[0], SizeToLong(x_shp.size()));
  if (axis_int == 0) {
    ShapeVector values_shape{ib->GetSize(indices)};
    if (x_shp.size() > 1) {
      (void)values_shape.insert(values_shape.end(), x_shp.begin() + 1, x_shp.end());
    }
    auto values = ib->Reshape(dout, values_shape);
    auto indices_new = ib->Reshape(indices, {values_shape[0]});
    auto row_tensor = ib->MakeTuple({indices_new, values, ib->Value<ShapeVector>(x_shp)});
    return {row_tensor, ib->OutZeros(indices), ib->OutZeros(axis)};
  }
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);
  if (out_shp.size() == 0) {
    dout = ib->ExpandDims(dout, -1);
  }
  if (ind_shp.size() == 0) {
    indices = ib->ExpandDims(indices, -1);
  }
  out_shp = ib->GetShape(dout);
  ind_shp = ib->GetShape(indices);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_int);
  auto values_transpose = ib->Transpose(dout, perm_1);
  auto params_grad = ib->UnsortedSegmentSum(values_transpose, indices, ib->Value<int64_t>(x_shp[LongToSize(axis_int)]));
  auto perm_2 = GenerateInverseIndex(x_shp, axis_int);
  params_grad = ib->Transpose(params_grad, perm_2);
  return {params_grad, ib->OutZeros(indices), ib->OutZeros(axis)};
});

DEF_PURE_SHAPE_CALC(g_sort_2)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto indices_shape = inputs[0];
    auto top_k_input_shape = inputs[1];
    auto indices_rank = indices_shape.size();
    auto top_k_input_rank = top_k_input_shape.size();
    if (indices_rank < 1 || top_k_input_rank < 1) {
      MS_LOG(EXCEPTION) << "For Sort, indices rank and top k rank should not less than 1, but got " << indices_rank
                        << " and " << top_k_input_rank;
    }
    auto ind_lastdim = indices_shape.at(indices_rank - 1);
    auto in_lastdim = top_k_input_shape.at(top_k_input_rank - 1);
    auto x_size = std::accumulate(top_k_input_shape.begin(), top_k_input_shape.end(), 1, std::multiplies<int64_t>());
    auto outer_dim = std::accumulate(indices_shape.begin(), indices_shape.end() - 1, 1, std::multiplies<int64_t>());
    return {top_k_input_shape, {-1, ind_lastdim}, {in_lastdim}, {x_size}, {outer_dim * in_lastdim}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    if (unknown_inputs.count(1) != 0) {
      return {-1, 2, 1, 1, 1};
    }
    auto top_k_input_shape = inputs[1];
    auto top_k_input_rank = top_k_input_shape.size();
    return {SizeToLong(top_k_input_rank), 2, 1, 1, 1};
  });

REG_BPROP_BUILDER("Sort").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto descending = GetValue<bool>(ib->GetAttr("descending"));
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto res1 = ib->ShapeCalc(std::make_shared<SortShapeCalc1>(axis), {input_x});
  auto k = res1[0];
  if (k->abstract()->isa<abstract::AbstractSequence>()) {
    if (k->input_type() == InputType::kConstant) {
      auto value = GetIntList(k);
      k = ib->Tensor(value.at(0), kInt64);
    } else {
      k = ib->TupleGetItem(k, 0);
    }
  }
  auto transposition = res1[1];
  auto invert_perm = res1[2];
  auto dvalue = ib->TupleGetItem(dout, 0);
  if (!descending) {
    input_x = ib->Neg(input_x);
    dvalue = ib->Neg(dvalue);
  }

  auto top_k_input = ib->Transpose(input_x, transposition);
  auto tmp = ib->Emit("TopK", {top_k_input, k}, {{"sorted", MakeValue(true)}});
  auto indices = ib->TupleGetItem(tmp, 1);
  auto res = ib->ShapeCalc(g_sort_2, {indices, top_k_input});
  auto indices_dtype = ib->GetDtype(indices);
  auto range_flatten_index =
    ib->Cast(ib->Range(ib->Value<int64_t>(0), ib->TupleGetItem(res[4], 0), ib->TupleGetItem(res[2], 0)), indices_dtype);
  range_flatten_index = ib->ExpandDims(range_flatten_index, -1);
  auto ind_2d = ib->Reshape(indices, res[1]);
  auto ind = ib->Reshape(ib->Add(ind_2d, range_flatten_index), {-1});

  dvalue = ib->Transpose(dvalue, invert_perm);
  auto ind_expand = ib->ExpandDims(ind, -1);
  auto scatter = ib->ScatterNd(ind_expand, ib->Reshape(dvalue, {-1}), res[3]);
  auto out_grad = ib->Reshape(scatter, res[0]);
  auto dx = ib->Transpose(out_grad, invert_perm);

  if (!descending) {
    dx = ib->Neg(dx);
  }
  return NodePtrList{dx};
});

REG_BPROP_BUILDER("SortExt").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);

  auto indices = ib->TupleGetItem(ib->GetInput(kIndex4), kIndex1);
  auto dout0 = ib->TupleGetItem(ib->GetInput(kIndex5), kIndex0);
  auto zeros = ib->Emit("ZerosLikeExt", {input, ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout0)))});
  auto res = ib->Emit("Scatter", {zeros, dim, indices, dout0, ib->EmitValue(MakeValue<int64_t>(0))});
  return {res, ib->OutZeros(dim), ib->OutZeros(ib->GetInput(kIndex2)), ib->OutZeros(ib->GetInput(kIndex3))};
});

REG_BPROP_BUILDER("Identity").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("Range").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("Arange").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Pack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);
REG_BPROP_BUILDER("Stack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);

REG_BPROP_BUILDER("ReverseV2").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ReverseV2", {dout, axis});
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Unstack").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Stack(dout, ib->GetAttr("axis"));
  return {dx};
});

REG_BPROP_BUILDER("StackExt").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  auto axis_node = ib->GetInput(kIndex1);
  auto input_shape = ib->GetShape(dout);
  if (input_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', 'x' can not be empty";
  }
  if (IsDynamicRank(input_shape)) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', DynamicRank is not supported";
  }
  auto axis_res = ops::GetScalarValue<int64_t>(axis_node->BuildValue());
  if (!axis_res.has_value()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', 'dim' can not be empty";
  }
  auto axis = axis_res.value();
  if (axis < 0) {
    axis += SizeToLong(input_shape.size());
  }
  auto num = input_shape[axis];
  auto ret = ib->Emit("Unstack", {dout}, {{"num", MakeValue(num)}, {"axis", MakeValue(axis)}});
  return {ret, ib->OutZeros(axis_node)};
});

REG_BPROP_BUILDER("Contiguous").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  // Transparent dout.
  return {ib->GetInput(kIndex2)};
});

REG_BPROP_BUILDER("StridedSlice").SetUnusedInputs({i0, i9}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto begin_mask = ib->GetInput(kIndex4);
  auto end_mask = ib->GetInput(kIndex5);
  auto ellipsis_mask = ib->GetInput(kIndex6);
  auto new_axis_mask = ib->GetInput(kIndex7);
  auto shrink_axis_mask = ib->GetInput(kIndex8);
  auto dout = ib->GetInput(kIndex10);
  auto x_shape_vec = ib->GetShape(x);

  NodePtr x_shape_node;
  if (IsDynamic(x_shape_vec)) {
    x_shape_node = ib->Shape(x);
  } else {
    x_shape_node = ib->EmitValue(MakeValue(x_shape_vec));
  }
  auto dx = ib->Emit("StridedSliceGrad", {dout, x_shape_node, begin, end, strides},
                     {{"begin_mask", begin_mask->BuildValue()},
                      {"end_mask", end_mask->BuildValue()},
                      {"ellipsis_mask", ellipsis_mask->BuildValue()},
                      {"new_axis_mask", new_axis_mask->BuildValue()},
                      {"shrink_axis_mask", shrink_axis_mask->BuildValue()}});
  auto dbegin = ib->OutZeros(begin);
  auto dend = ib->OutZeros(end);
  auto dstrides = ib->OutZeros(strides);
  return {dx,
          dbegin,
          dend,
          dstrides,
          ib->OutZeros(begin_mask),
          ib->OutZeros(end_mask),
          ib->OutZeros(ellipsis_mask),
          ib->OutZeros(new_axis_mask),
          ib->OutZeros(shrink_axis_mask)};
});

REG_BPROP_BUILDER("StridedSliceGrad").SetUnusedInputs({i0, i1, i5}).SetBody(BODYFUNC(ib) {
  auto shapex = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto strides = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto begin_mask = GetValue<int64_t>(ib->GetAttr("begin_mask"));
  auto end_mask = GetValue<int64_t>(ib->GetAttr("end_mask"));
  auto ellipsis_mask = GetValue<int64_t>(ib->GetAttr("ellipsis_mask"));
  auto new_axis_mask = GetValue<int64_t>(ib->GetAttr("new_axis_mask"));
  auto shrink_axis_mask = GetValue<int64_t>(ib->GetAttr("shrink_axis_mask"));
  return {
    ib->StridedSlice(dout, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask),
    ib->OutZeros(shapex), ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(strides)};
});

REG_BPROP_BUILDER("Eye").SetUnusedInputs({i0, i1, i3, i4}).SetBody(BODYFUNC(ib) {
  auto n = ib->GetInput(kIndex0);
  auto m = ib->GetInput(kIndex1);
  auto t = ib->GetInput(kIndex2);
  return {ib->OutZeros(n), ib->OutZeros(m), ib->OutZeros(t)};
});

REG_BPROP_BUILDER("Select").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto cond = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto y = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? ib->Select(cond, dout, ib->ZerosLike(dout)) : nullptr;
  auto dy = y->need_compute_grad_out() ? ib->Select(cond, ib->ZerosLike(dout), dout) : nullptr;
  auto ret = BinopGradSelect(ib, cond, x, y, dout, dx, dy);
  return {ib->OutZeros(cond), ret[kIndex1], ret[kIndex2]};
});

REG_BPROP_BUILDER("OnesLike").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ZerosLike").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("OnesLikeExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ZerosLikeExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

DEF_PURE_SHAPE_CALC(g_resize_nearest_neighbor)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto shape = inputs[0];
    ShapeVector res;
    for (size_t i = 2; i < shape.size(); ++i) {
      res.push_back(shape[i]);
    }
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    auto rank = SizeToLong(x.size());
    return {rank > 2 ? (rank - 2) : 0};
  });
REG_BPROP_BUILDER("ResizeNearestNeighbor").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto align_corners = ib->GetInput(kIndex2);
  auto half_pixel_centers = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape = ib->GetShape(x);
  NodePtr shape;
  if (!IsDynamic(x_shape)) {
    ShapeVector new_shape;
    for (size_t i = 2; i < x_shape.size(); i++) {
      new_shape.push_back(x_shape[i]);
    }
    shape = ib->EmitValue(MakeValue(new_shape));
  } else {
    shape = ib->ShapeCalc(g_resize_nearest_neighbor, {x})[0];
  }
  auto dx = ib->Emit("ResizeNearestNeighborGrad", {dout, shape, align_corners, half_pixel_centers}, {});
  return {dx, ib->OutZeros(ib->GetInput(kIndex1)), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("GatherNd").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shp = ib->Shape(x);
  return {ib->ScatterNd(ib->Cast(indices, kInt64), dout, shp), ib->OutZeros(indices)};
});

REG_BPROP_BUILDER("ScatterNd").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->OutZeros(indices), ib->GatherNd(dout, indices), ib->OutZeros(shape)};
});

REG_BPROP_BUILDER("ScatterNdUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = updates->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("ScatterNonAliasingAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("TensorScatterUpdate").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_grad =
    x->need_compute_grad_out() ? ib->TensorScatterUpdate(dout, indices, ib->ZerosLike(update)) : ib->OutZeros(x);
  auto updates_grad = x->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(x);
  return {x_grad, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("Flatten").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  if (IsDynamic(x_shape)) {
    return {ib->Reshape(dout, ib->Shape(x))};
  }
  return {ib->Reshape(dout, x_shape)};
});

REG_BPROP_BUILDER("FlattenExt").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto start = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shape = ib->Shape(x);
  return {ib->Reshape(dout, x_shape), ib->OutZeros(start), ib->OutZeros(end)};
});

REG_BPROP_BUILDER("Reshape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (!IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, shape_x);
  } else {
    dx = ib->Reshape(dout, ib->Shape(x));
  }
  return {dx, ib->OutZeros(shp)};
});

REG_BPROP_BUILDER("NonZero").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Argmax").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ArgMaxExt").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Argmin").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Diag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DiagPart", {dout})};
});

REG_BPROP_BUILDER("DiagPart").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("Diag", {dout})};
});

REG_BPROP_BUILDER("SpaceToBatch").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("BatchToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("SpaceToBatch", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("ReverseSequence").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto seq_lengths = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ReverseSequence", {dout, seq_lengths},
                     {{"batch_dim", ib->GetAttr("batch_dim")}, {"seq_dim", ib->GetAttr("seq_dim")}});
  return {dx, ib->OutZeros(seq_lengths)};
});

REG_BPROP_BUILDER("TensorScatterAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(update);
  return {dx, ib->OutZeros(indices), update_grad};
});

DEF_PURE_SHAPE_CALC(g_concat)
  .SetCalc([](const ShapeArray &inputs, const ElemPosIdx &pos_idx) -> ShapeArray {
    auto axis = inputs[pos_idx[1].front()][0];
    auto rank = inputs[0].size();
    auto axis_s = LongToSize(NormalizeAxis(axis, rank));
    ShapeArray input_shapes;
    input_shapes.reserve(pos_idx[0].size());
    for (auto idx : pos_idx[0]) {
      input_shapes.push_back(inputs[idx]);
    }
    return ConcatOffsetCal(input_shapes, axis_s);
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs,
               const ElemPosIdx &pos_idx) -> InferOutputInfo {
    auto x = inputs[0];
    auto x_rank = IsDynamicRank(x) ? abstract::TensorShape::kShapeDimAny : SizeToLong(x.size());
    if (unknown_inputs.count(0) != 0) {
      return std::make_pair(ShapeVector{x_rank}, true);
    }

    auto input_num = pos_idx[0].size();
    return std::make_pair(ShapeVector(input_num, x_rank), false);
  });
REG_BPROP_BUILDER("Concat").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);

  auto axis_node = ib->GetInput(kIndex1);

  auto x_abs = x->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  auto base_shape = x_abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto x_is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (!x_is_dyn_seq) {
    auto input_shapes = ib->GetShapes(x);
    if (input_shapes.empty()) {
      MS_EXCEPTION(ValueError) << "For gradient of 'Concat', 'x' can not be empty";
    }
    if (!std::any_of(input_shapes.cbegin(), input_shapes.cend(),
                     [](const std::vector<int64_t> &shape) { return IsDynamic(shape); })) {
      auto axis_res = ops::GetScalarValue<int64_t>(axis_node->BuildValue());
      if (axis_res.has_value()) {
        auto axis = axis_res.value();
        auto res = ConcatBpropStatic(ib, dout, input_shapes, axis);
        return {res[0], ib->OutZeros(axis_node)};
      }
    }

    auto concat_offset = ib->ShapeCalc(g_concat, {x, axis_node}, {1});
    auto input_nums = input_shapes.size();
    if (concat_offset.size() != input_nums) {
      MS_LOG(EXCEPTION) << "The number of ConcatOffset's ShapeCalc(" << concat_offset.size()
                        << ") is not equal to input(" << input_nums << ")!";
    }

    NodePtrList tuple_out;
    for (size_t i = 0; i < input_nums; ++i) {
      auto input = ib->Shape(ib->TupleGetItem(x, i));
      auto slice_out = ib->Emit(kSliceOpName, {dout, concat_offset[i], input});
      tuple_out.push_back(slice_out);
    }
    auto res = ib->MakeTuple(tuple_out);
    return {res, ib->OutZeros(axis_node)};
  }

  // Here the x is a dynamic sequence, so the infer out is a dynamic sequence too!
  auto concat_offset = ib->ShapeCalc(g_concat, {x, axis_node}, {1})[0];
  auto first_input = ib->Shape(ib->TupleGetItem(x, 0));
  auto offset = ib->TupleGetItem(concat_offset, 0);
  auto first_slice_out = ib->Emit(kSliceOpName, {dout, offset, first_input});
  auto res_list = ib->MakeList({first_slice_out});

  // Cannot use `auto i = ib->Tensor(1, kInt64);`.
  // If so, the i will be treat as a const Tensor but variable expected which will be not caught as a while body
  // parameter.
  // Here Emit a `ScalarToTensor` to make it a fake-variable-in-logit node.
  auto i = ib->Emit("ScalarToTensor", {ib->Value<int64_t>(1), ib->Value<int64_t>(kInt64->type_id())});
  auto len = ib->Emit("ScalarToTensor", {ib->Emit("sequence_len", {x}), ib->Value<int64_t>(kInt64->type_id())});
  auto while_body = [&x, &dout, &concat_offset, &i, &res_list](Emitter *e) -> NodePtrList {
    auto scalar_i = e->Emit("TensorToScalar", {i});
    auto input = e->Shape(e->Emit(kTupleGetItemOpName, {x, scalar_i}));
    auto offset = e->Emit(kTupleGetItemOpName, {concat_offset, scalar_i});
    auto slice_out = e->Emit(kSliceOpName, {dout, offset, input});
    auto new_list = e->Emit("ListAppend", {res_list, slice_out});
    auto new_i = e->Emit("Add", {i, e->Tensor(1, kInt64)});
    return {x, dout, concat_offset, new_i, new_list};
  };
  auto cond = ib->Less(i, len);
  auto while_block = ib->While(cond, while_body, {x, dout, concat_offset, i, res_list});
  // The `res_list` is a list, it should return a Tuple type rigorously.
  // Because there are no ListToTuple, and list type is ok for now.....
  auto dyn_list = ib->TupleGetItem(while_block, kIndex4);
  return {dyn_list, ib->OutZeros(axis_node)};
});

REG_BPROP_BUILDER("Mvlgamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MvlgammaGrad", {dout, x}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("TensorScatterDiv").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto in_grad = x->need_compute_grad_out() ? ib->Emit("TensorScatterDiv", {dout, indices, update}) : ib->OutZeros(x);
  NodePtr update_grad = nullptr;
  if (update->need_compute_grad_out()) {
    auto gather_update = ib->GatherNd(dout, indices);
    auto gather_x = ib->GatherNd(x, indices);
    auto mul_result = ib->Mul(update, update);
    auto neg_result = ib->Emit("Neg", {mul_result});
    update_grad = ib->Mul(gather_update, (ib->Div(gather_x, neg_result)));
  } else {
    update_grad = ib->OutZeros(update);
  }
  return {in_grad, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterSub").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad =
    update->need_compute_grad_out() ? ib->Emit("Neg", {ib->GatherNd(dout, indices)}) : ib->OutZeros(update);
  return {dx, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterMul").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? ib->Emit("TensorScatterMul", {dout, indices, update}) : ib->OutZeros(x);
  NodePtr d_update = nullptr;
  if (update->need_compute_grad_out()) {
    auto gather_update = ib->GatherNd(dout, indices);
    auto gather_x = ib->GatherNd(x, indices);
    d_update = ib->Mul(gather_x, gather_update);
  } else {
    d_update = ib->OutZeros(update);
  }
  return {dx, ib->OutZeros(indices), d_update};
});

REG_BPROP_BUILDER("TensorScatterMax").SetBody(TensorScatterPossibleReplacement);
REG_BPROP_BUILDER("TensorScatterMin").SetBody(TensorScatterPossibleReplacement);

REG_BPROP_BUILDER("IndexFill").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto indices = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto zero_value = ib->ZerosLike(value);
  auto x_grad = ib->Emit("IndexFill", {dout, dim, indices, zero_value});
  return {x_grad, ib->OutZeros(dim), ib->OutZeros(indices), zero_value};
});

REG_BPROP_BUILDER("UnsortedSegmentSum").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {GatherDropNegatives(ib, dout, segment_ids, nullptr, nullptr)[0], ib->OutZeros(segment_ids),
          ib->OutZeros(num_segments)};
});

REG_BPROP_BUILDER("UnsortedSegmentMin").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentMax").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentProd").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);

  NodePtr is_zero = nullptr;
  auto x_dtype = ib->GetDtype(x);
  MS_EXCEPTION_IF_NULL(x_dtype);
  auto x_dtype_id = x_dtype->type_id();
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'UnsortedSegmentProd', complex number is not supported for gradient currently.";
  }
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    is_zero = ib->Equal(x, ib->Tensor(0, x_dtype));
  } else {
    is_zero = ib->Equal(ib->Cast(x, kFloat32), ib->Tensor(0, kFloat32));
  }

  auto num_zero = ib->UnsortedSegmentSum(ib->Cast(is_zero, kInt32), segment_ids, num_segments);
  auto grad = ib->Select(ib->Greater(num_zero, ib->Tensor(1, ib->GetDtype(num_zero))), ib->ZerosLike(dout), dout);
  NodePtr non_zero_data = nullptr;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    non_zero_data = ib->Select(is_zero, ib->OnesLike(x), x);
  } else {
    auto temp_var = ib->OnesLike(ib->Cast(x, kFloat32));
    non_zero_data = ib->Select(is_zero, ib->Cast(temp_var, x_dtype_id), x);
  }
  auto non_zero_prod = ib->Emit("UnsortedSegmentProd", {non_zero_data, segment_ids, num_segments});
  auto zero_clipped_indices = ib->Maximum(segment_ids, ib->ZerosLike(segment_ids));
  auto gathered_prod = ib->Gather(out, zero_clipped_indices, 0);
  auto gathered_non_zero_prod = ib->Gather(non_zero_prod, zero_clipped_indices, 0);

  NodePtr prod_divided_by_x = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    prod_divided_by_x = ib->RealDiv(ib->Cast(gathered_prod, kFloat32), ib->Cast(x, kFloat32));
  } else {
    prod_divided_by_x = ib->RealDiv(gathered_prod, x);
  }
  auto partial_derivative =
    ib->Select(is_zero, gathered_non_zero_prod, ib->Cast(prod_divided_by_x, ib->GetDtype(gathered_non_zero_prod)));

  auto temp_outs = GatherDropNegatives(ib, grad, segment_ids, zero_clipped_indices, nullptr);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grad = temp_outs[0];
  NodePtr dx = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    auto temp_dx = ib->Mul(ib->Cast(gathered_grad, kFloat32), ib->Cast(partial_derivative, kFloat32));
    dx = ib->Cast(temp_dx, x_dtype);
  } else {
    dx = ib->Mul(gathered_grad, partial_derivative);
  }

  return {dx, ib->OutZeros(segment_ids), ib->OutZeros(num_segments)};
});

REG_BPROP_BUILDER("SpaceToBatchND").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("BatchToSpaceND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpaceND").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SpaceToBatchND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("BroadcastTo").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  auto dout_shape = ib->GetShape(dout);

  bool input_dynamic = IsDynamic(x_shape) || IsDynamic(dout_shape);
  if (!input_dynamic && x_shape == dout_shape) {
    return {dout, ib->OutZeros(ib->GetInput(kIndex1))};
  }

  auto x_shape_node = ib->Shape(x);
  auto broadcast_axes = ib->BroadcastGradientArgs(dout, x);
  MS_EXCEPTION_IF_CHECK_FAIL(!broadcast_axes.empty(), "BroadcastGradientArgs out should not be empty!");
  auto reduction_axes = broadcast_axes[kIndex1];
  NodePtr reduced_grad = nullptr;

  reduced_grad = ib->ReduceSum(dout, reduction_axes, true, true);
  auto dx = ib->Reshape(reduced_grad, x_shape_node);

  return {dx, ib->OutZeros(ib->GetInput(kIndex1))};
});

REG_BPROP_BUILDER("SpaceToDepth").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DepthToSpace", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("DepthToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("SpaceToDepth", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("ScatterMax").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("ScatterMin").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("ScatterUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("NormalTensorTensor").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalTensorFloat").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalFloatTensor").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalFloatFloat").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("UniformExt").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ScatterAddExt").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto indices = ib->GetInput(kIndex2);
  auto update = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  NodePtr x_grad = nullptr;
  x_grad = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->GatherD(dout, axis, indices) : ib->OutZeros(update);
  return {x_grad, ib->OutZeros(axis), ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("NormalizeSlice").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalizeDimIndex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto data = ib->GetInput(kIndex0);
  return {ib->ZerosLike(data)};
});

REG_BPROP_BUILDER("NormalizeTupleIndex").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("GetSqueezeSliceShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto data = ib->GetInput(kIndex0);
  return {ib->ZerosLike(data)};
});

REG_BPROP_BUILDER("EllipsisToSlice").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("RemakeTupleIndex")
  .SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10})
  .SetBody(ReturnZeros);

REG_BPROP_BUILDER("GetTupleIndexInfo")
  .SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10})
  .SetBody(ReturnZeros);

REG_BPROP_BUILDER("RemoveExpandedDims").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("SliceToIndices").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Fills").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Cast").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr dx;
  if (dout->abstract()->isa<abstract::AbstractRowTensor>()) {
    auto row_tensor_values = ib->Emit("RowTensorGetValues", {dout});
    auto value = ib->Cast(row_tensor_values, x_dtype);
    auto indices = ib->Emit("RowTensorGetIndices", {dout});
    auto dense_shape = ib->Emit("RowTensorGetDenseShape", {dout});
    dx = ib->Emit("MakeRowTensor", {indices, value, dense_shape});
  } else {
    dx = ib->Cast(dout, x_dtype);
  }
  auto abs = x->abstract();
  if (!(abs->isa<abstract::AbstractScalar>())) {
    return {dx, ib->OutZeros(t)};
  }
  auto dx_ = ib->Emit("TensorToScalar", {dx});
  return {dx_, ib->OutZeros(t)};
});

REG_BPROP_BUILDER("ExpandDims").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, ib->Shape(x));
  } else {
    dx = ib->Reshape(dout, shape_x);
  }
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Squeeze").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shapex = ib->GetShape(x);
  if (IsDynamic(shapex)) {
    return {ib->Reshape(dout, ib->Shape(x))};
  }
  return {ib->Reshape(dout, shapex)};
});

REG_BPROP_BUILDER("Padding").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shp = ib->GetShape(x);
  if (!IsDynamic(shp)) {
    std::vector<int64_t> begin(shp.size(), 0);
    auto dx = ib->Slice(dout, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(shp));
    return {dx};
  }

  auto shape_node = ib->Shape(x);
  auto begin_node = ib->ZerosLike(shape_node);
  return {ib->Slice(dout, begin_node, shape_node)};
});

DEF_PURE_SHAPE_CALC(g_transpose)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto perm = inputs[0];
    std::vector<int64_t> new_perm;
    (void)std::transform(perm.begin(), perm.end(), std::back_inserter(new_perm),
                         [&perm](const int64_t v) { return v >= 0 ? v : v + SizeToLong(perm.size()); });
    auto res_perm = InvertPermutation(new_perm);
    return {res_perm};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    return {SizeToLong(x.size())};
  });
REG_BPROP_BUILDER("Transpose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res_perm = ib->ShapeCalc(g_transpose, {perm}, {0})[0];
  return {ib->Transpose(dout, res_perm), ib->OutZeros(perm)};
});

REG_BPROP_BUILDER("Slice").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto size = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("SliceGrad", {dout, x, begin, size});
  return {dx, ib->OutZeros(begin), ib->OutZeros(size)};
});

REG_BPROP_BUILDER("Split").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->GetInput(kIndex1);
  auto output_num = ib->GetInput(kIndex2);
  auto axis_ptr = axis->BuildValue();
  MS_EXCEPTION_IF_NULL(axis_ptr);
  auto axis_value = GetValue<int64_t>(axis_ptr);
  auto dx = ib->Concat(dout, axis_value);
  return {dx, ib->OutZeros(axis), ib->OutZeros(output_num)};
});

REG_BPROP_BUILDER("SplitTensor").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto split_int = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout, axis});
  return {dx, ib->OutZeros(split_int), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("SplitWithSize").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto split_sections = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout, axis});
  return {dx, ib->OutZeros(split_sections), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Chunk").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto chunks = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("Concat", {dout, axis});
  return {dx, ib->OutZeros(chunks), ib->OutZeros(axis)};
});

DEF_PURE_SHAPE_CALC(g_slice_ext)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    auto axis = inputs.at(1);
    auto begin = inputs.at(2);
    auto end = inputs.at(3);

    MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
    auto axis_value = axis[0];
    MS_EXCEPTION_IF_CHECK_FAIL(begin.size() == 1, "begin should be a scalar.");
    auto begin_value = begin[0];
    MS_EXCEPTION_IF_CHECK_FAIL(end.size() == 1, "end should be a scalar.");
    auto end_value = end[0];

    axis_value = axis_value < 0 ? axis_value + x_shape.size() : axis_value;
    auto length_value = end_value - begin_value;
    begin_value = begin_value < 0 ? begin_value + x_shape[axis_value] : begin_value;
    end_value = begin_value + length_value;

    auto begin_shape = x_shape;
    begin_shape[axis_value] = begin_value;
    auto end_shape = x_shape;
    end_shape[axis_value] = end_shape[axis_value] - end_value;

    return {begin_shape, end_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    auto axis = inputs.at(1);
    auto begin = inputs.at(2);
    auto end = inputs.at(3);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(axis) || IsDynamicRank(begin) ||
        IsDynamicRank(end)) {
      return {-1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, size};
  });

REG_BPROP_BUILDER("SliceExt").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto step = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto res = ib->ShapeCalc(g_slice_ext, {x, axis, begin, end}, {1, 2, 3});
  auto dx =
    ib->Emit(kConcatOpName, {ib->MakeTuple({ib->Emit("Zeros", {res[0], ib->Value<int64_t>(ib->GetDtypeId(dout))}), dout,
                                            ib->Emit("Zeros", {res[1], ib->Value<int64_t>(ib->GetDtypeId(dout))})}),
                             axis});

  return {dx, ib->OutZeros(axis), ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(step)};
});

DEF_PURE_SHAPE_CALC(g_tile)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    // {x_shape, dims}
    auto r_shape = TileShape(inputs.at(1), inputs.at(0));
    ShapeVector axis;
    size_t axis_sz = r_shape.size() / 2;
    axis.reserve(axis_sz);
    for (int64_t i = 0; i < static_cast<int64_t>(axis_sz); ++i) {
      axis.push_back(i * 2);
    }
    return {r_shape, axis};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    auto dims = inputs.at(1);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(dims)) {
      return {-1, -1};
    }
    auto x_sz = static_cast<int64_t>(x.size());
    auto multiples_sz = static_cast<int64_t>(dims.size());
    auto max_sz = x_sz > multiples_sz ? x_sz : multiples_sz;
    return {2 * max_sz, max_sz};
  });
REG_BPROP_BUILDER("Tile").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto input_multiples = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto calc_res = ib->ShapeCalc(g_tile, {x, input_multiples}, {1});
  auto r_shape = calc_res[0];
  auto axis = calc_res[1];
  auto dout_reshaped = ib->Reshape(dout, r_shape);
  NodePtr dx;
  auto need_reduce = ib->NeedReduce(r_shape, axis, false);
  if (need_reduce.first) {
    dx = ib->ReduceSum(dout_reshaped, axis);
  } else {
    dx = ib->Reshape(dout_reshaped, ib->TensorToTuple(need_reduce.second));
  }
  auto shape_x = ib->Shape(x);
  dx = ib->Reshape(dx, shape_x);
  return {dx, ib->OutZeros(input_multiples)};
});

REG_BPROP_BUILDER("Gather").SetUnusedInputs({i0, i4}).SetBody(BinopGather);

REG_BPROP_BUILDER("Fill").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("SelectView").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto idx = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto out_shp = ib->GetShape(dout);
  auto indices = ib->Tensor(GetIntValue(idx), kInt64);
  auto ori_indices = indices;  // indices may be changed latter.
  auto ind_shp = ib->GetShape(indices);
  int64_t axis_v = 0;
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(axis->abstract());
  auto axis_tmp = axis->BuildValue();
  MS_EXCEPTION_IF_NULL(axis_tmp);
  axis_v = CheckRange(GetIntValue(axis), SizeToLong(x_shp.size()));
  int64_t batch_dims = 0;

  if (out_shp.empty()) {
    dout = ib->ExpandDims(dout, -1);
  } else {
    dout = ib->ExpandDims(dout, axis_v);
  }

  auto is_axis_mutable = IsMutable(axis);
  if ((!is_axis_mutable && (IsDynamicRank(x_shp) || IsDynamicRank(ind_shp) || IsDynamicRank(out_shp))) ||
      (is_axis_mutable && (IsDynamic(x_shp) || IsDynamic(ind_shp) || IsDynamic(out_shp)))) {
    auto batch_dims_tensor = ib->Tensor(batch_dims, kInt64);
    if (ind_shp.empty()) {
      indices = ib->ExpandDims(indices, -1);
      auto out_shp1 = ib->ShapeCalc(g_regenerate_output, {x, indices, axis, batch_dims_tensor}, {kIndex2, kIndex3})[0];
      dout = ib->Reshape(dout, out_shp1);
    }

    // Calculate perm.
    auto perms = ib->ShapeCalc(g_perms, {x, dout, indices, axis, batch_dims_tensor}, {kIndex3, kIndex4});
    const size_t perm_num = 2;
    MS_EXCEPTION_IF_CHECK_FAIL(perms.size() == perm_num, "Perms number should be 2 for gradient of Gather.");
    auto perm_1 = perms[0];
    auto perm_2 = perms[1];
    auto values_transpose = ib->Transpose(dout, perm_1);
    NodePtr x_grad = nullptr;
    if (batch_dims > 0) {
      x_grad = CalBatchGather(ib, values_transpose, indices, x, axis_v, batch_dims);
    } else {
      auto num_segment = CalcNumSegment(ib, x, axis);
      x_grad = ib->UnsortedSegmentSum(values_transpose, indices, num_segment);
    }
    x_grad = ib->Transpose(x_grad, perm_2);
    return {x_grad, ib->OutZeros(ori_indices), ib->OutZeros(axis)};
  }

  if (ind_shp.empty()) {
    indices = ib->ExpandDims(indices, -1);
    ind_shp = ib->GetShape(indices);
    auto out_shp1 = RegenerateOutputShape(x_shp, ind_shp, axis_v);
    dout = ib->Reshape(dout, out_shp1);
  }

  out_shp = ib->GetShape(dout);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_v, batch_dims);
  auto values_transpose = ib->Transpose(dout, perm_1);
  NodePtr x_grad = nullptr;
  auto num_segment = CalcNumSegment(ib, x, axis);
  x_grad = ib->UnsortedSegmentSum(values_transpose, indices, num_segment);
  auto perm_2 = GenerateInverseIndex(x_shp, axis_v, batch_dims);
  auto params_grad = ib->Transpose(x_grad, perm_2);
  return {params_grad, ib->OutZeros(ori_indices), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("MatrixBandPart").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto lower = ib->GetInput(kIndex1);
  auto upper = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto part = ib->Emit("MatrixBandPart", {dout, lower, upper});
  return {part, ib->OutZeros(lower), ib->OutZeros(upper)};
});

REG_BPROP_BUILDER("MatrixDiagV3").SetUnusedInputs({i0, i2, i3, i4, i5}).SetBody(BODYFUNC(ib) {
  auto k = ib->GetInput(kIndex1);
  auto num_rows = ib->GetInput(kIndex2);
  auto num_cols = ib->GetInput(kIndex3);
  auto padding_value = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto part = ib->MatrixDiagPartV3(dout, k, ib->Tensor(0, ib->GetDtype(dout)), ib->GetAttr("align"));
  return {part, ib->OutZeros(k), ib->OutZeros(num_rows), ib->OutZeros(num_cols), ib->OutZeros(padding_value)};
});

REG_BPROP_BUILDER("MatrixDiagPartV3").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto align = ib->GetAttr("align");
  auto x = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto padding_value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shape = ib->GetShape(x);
  bool is_dynamic_case = IsDynamicRank(x_shape);
  ShapeVector sub_shape;
  if (!is_dynamic_case) {
    size_t begin = (x_shape.size() < 2) ? 0 : (x_shape.size() - 2);
    for (; begin < x_shape.size(); ++begin) {
      sub_shape.push_back(x_shape[begin]);
    }
    is_dynamic_case = IsDynamic(sub_shape);
  }

  NodePtr diag = nullptr;
  if (!is_dynamic_case) {
    if (sub_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For gradient of MatrixDiagPartV3, rank should be greater than 2";
    }
    auto row = x_shape[x_shape.size() - 2];
    auto col = x_shape[x_shape.size() - 1];
    diag = ib->Emit("MatrixDiagV3",
                    {dout, k, ib->Tensor(row, kInt32), ib->Tensor(col, kInt32), ib->Tensor(0, ib->GetDtype(dout))},
                    {{"align", align}});
  } else {
    diag = ib->MatrixSetDiagV3(ib->ZerosLike(x), dout, k, align);
  }
  return {diag, ib->OutZeros(k), ib->OutZeros(padding_value)};
});

REG_BPROP_BUILDER("MatrixSetDiagV3").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  auto align = ib->GetAttr("align");
  auto x = ib->GetInput(kIndex0);
  auto diagonal = ib->GetInput(kIndex1);
  auto k = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto diagonal_grad = diagonal->need_compute_grad_out()
                         ? ib->MatrixDiagPartV3(dout, k, ib->Tensor(0, ib->GetDtype(dout)), align)
                         : ib->OutZeros(diagonal);
  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    auto diagonal_shape = ib->GetShape(diagonal);
    auto dout_type = ib->GetDtypeId(dout);
    if (IsDynamic(diagonal_shape)) {
      auto diagonal_temp = ib->Cast(diagonal, dout_type);
      dx = ib->MatrixSetDiagV3(dout, ib->ZerosLike(diagonal_temp), k, align);
    } else {
      dx = ib->MatrixSetDiagV3(dout, ib->Fill(static_cast<int64_t>(0), diagonal_shape, dout_type), k, align);
    }
  } else {
    dx = ib->OutZeros(x);
  }
  return {dx, diagonal_grad, ib->OutZeros(k)};
});

REG_BPROP_BUILDER("LogNormalReverse").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Shape").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Rank").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("DynamicShape").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("TensorShape").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("DType").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Size").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("StridedSliceV2").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape_vec = ib->GetShape(x);
  NodePtr x_shape;
  if (IsDynamic(x_shape_vec)) {
    x_shape = ib->Shape(x);
  } else {
    x_shape = ib->Tensor(x_shape_vec);
  }
  auto dx = ib->Emit("StridedSliceV2Grad", {x_shape, begin, end, strides, dout},
                     {{"begin_mask", ib->GetAttr("begin_mask")},
                      {"end_mask", ib->GetAttr("end_mask")},
                      {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                      {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                      {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}});
  return {dx, ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(strides)};
});

REG_BPROP_BUILDER("MaskedFill").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto input_data = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dmask = ib->OutZeros(mask);
  mask = ib->Cast(mask, kFloat32);
  dout = ib->Cast(dout, kFloat32);
  NodePtr dinput =
    input_data->need_compute_grad_out() ? ib->Mul(dout, ib->Sub((ib->Tensor(1, ib->GetDtype(mask))), mask)) : nullptr;
  NodePtr dvalue = value->need_compute_grad_out() ? ib->Mul(dout, mask) : nullptr;
  auto bout = BinopGradCommon(ib, input_data, mask, dinput, dvalue);

  dinput = input_data->need_compute_grad_out() ? ib->Cast(bout[0], ib->GetDtype(input_data)) : ib->OutZeros(dinput);
  if (value->need_compute_grad_out()) {
    auto dvalue_shape = dvalue->shape();
    if (IsDynamicRank(dvalue_shape)) {
      auto dvalue_rank = ib->Shape(ib->Shape(dvalue, true), true);
      auto axis_node = ib->Range(ib->TensorToScalar(dvalue_rank));
      dvalue = ib->ReduceSum(bout[1], axis_node);
    } else {
      dvalue = ib->ReduceSum(bout[1]);
    }
    dvalue = ib->Cast(dvalue, ib->GetDtype(value));
  } else {
    dvalue = ib->OutZeros(value);
  }
  return {dinput, dmask, dvalue};
});

REG_BPROP_BUILDER("Coalesce").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto d1 = ib->TupleGetItem(dout, 0);
  auto d2 = ib->TupleGetItem(dout, 1);
  auto d3 = ib->TupleGetItem(dout, 2);
  return {d1, d2, d3};
});

REG_BPROP_BUILDER("ConjugateTranspose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp_perm = GetIntList(perm);
  auto tmp_perm_sz = SizeToLong(tmp_perm.size());
  std::vector<int64_t> new_perm;
  (void)std::transform(tmp_perm.begin(), tmp_perm.end(), std::back_inserter(new_perm),
                       [&tmp_perm, tmp_perm_sz](const int64_t v) { return v >= 0 ? v : v + tmp_perm_sz; });
  auto res_perm = InvertPermutation(new_perm);
  return {ib->Emit("ConjugateTranspose", {dout, ib->Value<ShapeVector>(res_perm)}), ib->OutZeros(perm)};
});

REG_BPROP_BUILDER("Triu").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto diagonal = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Triu", {dout, diagonal});
  return {dx, ib->OutZeros(diagonal)};
});

REG_BPROP_BUILDER("CheckNumerics").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("CheckNumerics", {dout})};
});

REG_BPROP_BUILDER("IdentityN").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("ResizeNearestNeighborV2").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto align_corners = ib->GetInput(kIndex2);
  auto half_pixel_centers = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto grad_in_size = ib->ShapeCalc(std::make_shared<ResizeNearestNeighborV2ShapeCalc>(true), {x})[0];
  if (grad_in_size->input_type() == InputType::kConstant) {
    grad_in_size = ib->Value<ShapeVector>(GetIntList(grad_in_size));
  }
  auto dx = ib->Emit("ResizeNearestNeighborV2Grad", {dout, grad_in_size, align_corners, half_pixel_centers});
  return {dx, ib->OutZeros(grad_in_size), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("Tril").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Tril", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("SegmentSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_type = ib->GetDtype(dout);
  std::set<TypePtr> type_list = {kInt8, kInt16, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  if (CheckType(dout_type, type_list)) {
    dout = ib->Cast(dout, kInt32);
  }
  if (dout_type->type_id() == TypeId::kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  return {ib->Cast(ib->Gather(dout, segment_ids, ib->EmitValue(MakeValue<int64_t>(0))), dout_type),
          ib->OutZeros(segment_ids)};
});

REG_BPROP_BUILDER("EmbeddingLookup").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto offset = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto offset_v = GetIntValue(offset);
  auto new_indices = ib->Sub(indices, ib->Tensor(offset_v, ib->GetDtype(indices)));
  auto indices_size = ib->GetSize(new_indices);
  ShapeVector new_indices_shape;
  ShapeVector x_shp_tail;
  ShapeVector actual_dout_shape;
  if (indices_size > 0) {
    new_indices_shape.push_back(indices_size);
    new_indices = ib->Reshape(new_indices, new_indices_shape);
  }
  int64_t x_rank = static_cast<int64_t>(x_shp.size());
  auto start1 = x_rank <= 1 ? x_shp.end() : x_shp.begin() + 1;
  (void)std::copy(start1, x_shp.end(), std::back_inserter(x_shp_tail));
  (void)std::copy(new_indices_shape.begin(), new_indices_shape.end(), std::back_inserter(actual_dout_shape));
  (void)std::copy(x_shp_tail.begin(), x_shp_tail.end(), std::back_inserter(actual_dout_shape));
  auto actual_dout = ib->Reshape(dout, actual_dout_shape);
  return {ib->MakeTuple({new_indices, actual_dout, ib->Value<ShapeVector>(x_shp)}), ib->OutZeros(indices),
          ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("MaskedSelect").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MaskedSelectGrad", {x, mask, dout});
  return {dx, ib->OutZeros(mask)};
});

REG_BPROP_BUILDER("SplitV").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto split_dim = GetValue<int64_t>(ib->GetAttr("split_dim"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Concat(dout, split_dim);
  return {dx};
});

REG_BPROP_BUILDER("Col2Im").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto ksizes = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilation"));
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("stride"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("padding"));
  auto output_size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Im2Col", {dout},
                     {{"ksizes", MakeValue(ksizes)},
                      {"dilations", MakeValue(dilations)},
                      {"strides", MakeValue(strides)},
                      {"padding_mode", MakeValue("CALCULATED")},
                      {"pads", MakeValue(pads)}});
  return {dx, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("ExtractVolumePatches").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto ksize = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto ksize_d = ksize.at(2);
  auto ksize_h = ksize.at(3);
  auto ksize_w = ksize.at(4);
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  auto x_n = x_shape.at(0);
  auto x_c = x_shape.at(1);
  auto x_d = x_shape.at(2);
  auto x_h = x_shape.at(3);
  auto x_w = x_shape.at(4);
  auto x_indices_num = 1 + ((x_d * x_h) * x_w);
  auto x_idx = ib->Tensor(Range(1, x_indices_num), kFloat16);
  x_idx = ib->Reshape(x_idx, {1, 1, x_d, x_h, x_w});
  auto x_idx_patched = ib->Emit("ExtractVolumePatches", {x_idx},
                                {{"kernel_size", ib->GetAttr("kernel_size")},
                                 {"strides", ib->GetAttr("strides")},
                                 {"padding", ib->GetAttr("padding")}});
  x_idx_patched = ib->Transpose(x_idx_patched, {0, 2, 3, 4, 1});
  x_idx_patched = ib->Cast(x_idx_patched, kInt32);
  auto out_shape = ib->GetShape(out);
  auto out_d = out_shape.at(2);
  auto out_h = out_shape.at(3);
  auto out_w = out_shape.at(4);
  auto out_indices_num = ((((out_d * out_h) * out_w) * ksize_d) * ksize_h) * ksize_w;
  auto out_idx = ib->Tensor(Range(0, out_indices_num), kInt32);
  out_idx = ib->Reshape(out_idx, {1, out_d, out_h, out_w, (ksize_d * ksize_h) * ksize_w});
  auto idx_tensor = ib->Concat({ib->ExpandDims(x_idx_patched, -1), ib->ExpandDims(out_idx, -1)}, -1);
  auto idx_map = ib->Reshape(idx_tensor, {-1, 2});
  std::vector<int64_t> sp_shape = {x_indices_num, out_indices_num};
  std::vector<int64_t> ones(out_indices_num, 1);
  auto sp_mat_full = ib->ScatterNd(idx_map, ib->Tensor(ones, ib->GetDtype(dout)), ib->Value<ShapeVector>(sp_shape));
  auto sp_tensor = ib->Slice(sp_mat_full, ib->Value<ShapeVector>({1, 0}),
                             ib->Value<ShapeVector>({x_indices_num - 1, out_indices_num}));
  auto grad = ib->Transpose(dout, {0, 2, 3, 4, 1});
  grad = ib->Reshape(grad, {x_n, out_d, out_h, out_w, ksize_d, ksize_h, ksize_w, x_c});
  auto grad_expended = ib->Transpose(grad, {1, 2, 3, 4, 5, 6, 0, 7});
  auto grad_flat = ib->Reshape(grad_expended, {-1, x_n * x_c});
  auto jac = ib->MatMul(sp_tensor, grad_flat, false, false);
  auto dx = ib->Reshape(jac, {x_d, x_h, x_w, x_n, x_c});
  dx = ib->Transpose(dx, {3, 4, 0, 1, 2});
  return {dx};
});

REG_BPROP_BUILDER("AffineGrid").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == "GPU") {
    auto align_corners = GetValue<bool>(ib->GetAttr("align_corners"));
    auto output_size = GetIntList(ib->GetInput(kIndex1));
    auto dout = ib->GetInput(kIndex3);
    auto start = ib->Tensor(-1, kFloat32);
    auto stop = ib->Tensor(1, kFloat32);
    auto zero = ib->Tensor(0, kFloat32);
    constexpr int64_t c0 = 0;
    constexpr int64_t c1 = 1;
    constexpr int64_t c2 = 2;
    constexpr int64_t c3 = 3;
    constexpr int64_t c4 = 4;
    ShapeVector perm1{c1, c0};
    ShapeVector perm2{c0, c2, c1};
    if (output_size.size() == kDim5) {
      const auto n_value = output_size[kIndex0];
      const auto d_value = output_size[kIndex2];
      const auto h_value = output_size[kIndex3];
      const auto w_value = output_size[kIndex4];
      auto vecx = (w_value != 1) ? ib->LinSpace(start, stop, ib->Value(w_value)) : zero;
      auto vecy = (h_value != 1) ? ib->LinSpace(start, stop, ib->Value(h_value)) : zero;
      auto vecz = (d_value != 1) ? ib->LinSpace(start, stop, ib->Value(d_value)) : zero;
      if (!align_corners) {
        vecx = (vecx * ib->Tensor(w_value - 1, kFloat32)) / ib->Tensor(w_value, kFloat32);
        vecy = (vecy * ib->Tensor(h_value - 1, kFloat32)) / ib->Tensor(h_value, kFloat32);
        vecz = (vecz * ib->Tensor(d_value - 1, kFloat32)) / ib->Tensor(d_value, kFloat32);
      }
      auto out = (h_value * d_value != 1) ? ib->Tile(vecx, {h_value * d_value, 1}) : vecx;
      auto one = ib->Reshape(out, {h_value * w_value * d_value, 1});
      out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
      out = ib->Transpose(out, perm1);
      if (d_value != 1) {
        out = ib->Tile(out, {d_value, 1});
      }
      auto two = ib->Reshape(out, {h_value * w_value * d_value, 1});
      out = (w_value * h_value != 1) ? ib->Tile(vecz, {w_value * h_value, 1}) : ib->ExpandDims(vecz, 0);
      out = ib->Transpose(out, perm1);
      auto tre = ib->Reshape(out, {h_value * w_value * d_value, 1});
      auto fou = ib->OnesLike(tre);
      auto output = ib->Concat({one, two, tre, fou}, 1);
      output = ib->Transpose(output, perm1);
      if (n_value != 1) {
        output = ib->Tile(output, {n_value, 1});
      }
      output = ib->Reshape(output, {n_value, c4, h_value * w_value * d_value});
      dout = ib->Reshape(dout, {n_value, d_value * h_value * w_value, c3});
      dout = ib->Cast(dout, kFloat32);
      auto dtheta = ib->BatchMatMul(output, dout);
      dtheta = ib->Transpose(dtheta, perm2);
      return {dtheta, tre};
    } else if (output_size.size() == kDim4) {
      auto x_shape = ib->GetShape(dout);
      const auto n_value = x_shape[kIndex0];
      const auto h_value = x_shape[kIndex1];
      const auto w_value = x_shape[kIndex2];
      auto vecx = (w_value != 1) ? ib->LinSpace(start, stop, ib->Value(w_value)) : zero;
      auto vecy = (h_value != 1) ? ib->LinSpace(start, stop, ib->Value(h_value)) : zero;
      if (!align_corners) {
        vecx = (vecx * ib->Tensor(w_value - 1, kFloat32)) / ib->Tensor(w_value, kFloat32);
        vecy = (vecy * ib->Tensor(h_value - 1, kFloat32)) / ib->Tensor(h_value, kFloat32);
      }
      auto out = (h_value != 1) ? ib->Tile(vecx, {h_value, 1}) : vecx;
      auto one = ib->Reshape(out, {h_value * w_value, 1});
      out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
      out = ib->Transpose(out, perm1);
      auto two = ib->Reshape(out, {h_value * w_value, 1});
      auto tre = ib->OnesLike(two);
      auto output = ib->Concat({one, two, tre}, 1);
      output = ib->Transpose(output, perm1);
      output = ib->Tile(output, {n_value, 1});
      output = ib->Reshape(output, {n_value, c3, h_value * w_value});
      dout = ib->Reshape(dout, {n_value, h_value * w_value, c2});
      dout = ib->Cast(dout, kFloat32);
      auto dtheta = ib->BatchMatMul(output, dout);
      dtheta = ib->Transpose(dtheta, perm2);
      return {dtheta, tre};
    }
    MS_LOG(EXCEPTION) << "For op[" << ib->name() << "], the length of output_size should be 4 or 5, but got "
                      << output_size.size();
  } else {
    auto output_size = ib->GetInput(kIndex1);
    auto dout = ib->GetInput(kIndex3);
    auto dx = ib->Emit("AffineGridGrad", {dout, output_size}, {{"align_corners", ib->GetAttr("align_corners")}});
    return {dx, ib->OutZeros(output_size)};
  }
});

REG_BPROP_BUILDER("SegmentMax").SetBody(SegmentMinOrMaxGrad);
REG_BPROP_BUILDER("SegmentMin").SetBody(SegmentMinOrMaxGrad);

REG_BPROP_BUILDER("TensorScatterElements").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->GetAttr("axis");
  NodePtr x_grad = nullptr;
  if (x->need_compute_grad_out()) {
    x_grad = ib->Emit("TensorScatterElements", {dout, indices, ib->ZerosLike(update)},
                      {{"axis", axis}, {"reduction", ib->GetAttr("reduction")}});
  } else {
    x_grad = ib->OutZeros(x);
  }
  auto update_grad =
    update->need_compute_grad_out() ? ib->GatherD(dout, ib->EmitValue(axis), indices) : ib->OutZeros(update);
  return {x_grad, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("ScatterAddWithAxis").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetAttr("axis");
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dout_shape = ib->GetShape(dout);
  auto index_shape = ib->GetShape(indices);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  NodePtr update_grad = nullptr;
  if (update->need_compute_grad_out()) {
    if (dout_shape != index_shape) {
      ShapeVector slice_list(dout_shape.size(), 0);
      std::vector<ShapeVector> pad_list;
      pad_list.reserve(dout_shape.size());
      for (size_t i = 0; i < dout_shape.size(); i++) {
        (void)pad_list.emplace_back(ShapeVector{0, dout_shape[i] - index_shape[i]});
      }
      auto out_index = ib->Emit("Pad", {indices}, {{"paddings", MakeValue(pad_list)}});
      auto out_gather = ib->GatherD(dout, ib->EmitValue(axis), out_index);
      update_grad = ib->Slice(out_gather, ib->Value(slice_list), ib->Value(index_shape));
    } else {
      update_grad = ib->GatherD(dout, ib->EmitValue(axis), indices);
    }
  } else {
    update_grad = ib->OutZeros(update);
  }
  return {dx, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("CopyWithSlice").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  return {ib->OutZeros(dout), dout};
});

REG_BPROP_BUILDER("Expand").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_shape = ib->GetShape(dout);
  auto dshape = ib->OutZeros(shape);
  if (dout_shape.empty()) {
    return {ib->ReduceSum(dout), dshape};
  }
  auto x_shape = ib->GetShape(x);
  auto leading_dims = dout_shape.size() - x_shape.size();
  auto reduce_dims = Range(SizeToLong(leading_dims));
  for (size_t j = leading_dims; j < dout_shape.size(); ++j) {
    if (x_shape[j - leading_dims] == 1 && dout_shape[j] != 1) {
      reduce_dims.push_back(j);
    }
  }
  if (!reduce_dims.empty()) {
    dout = ib->ReduceSum(dout, reduce_dims, true);
  }
  auto dx = leading_dims > 0 ? ib->Reshape(dout, x_shape) : dout;
  return {dx, dshape};
});

DEF_PURE_SHAPE_CALC(g_segment_mean)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_rank = inputs.at(0).size();
    auto segment_ids_shape = inputs.at(1);
    ShapeVector ones_shape(segment_ids_shape.begin(), segment_ids_shape.end());
    if (x_rank < 1) {
      MS_LOG(EXCEPTION) << "For SegmentMean's gradient, the rank of input x should be greater or equal to one, but got "
                        << x_rank;
    }
    ShapeVector rank_shape(x_rank - 1, 1LL);
    (void)ones_shape.insert(ones_shape.end(), rank_shape.begin(), rank_shape.end());
    return {ones_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    auto segment_ids = inputs.at(1);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(segment_ids)) {
      return {-1};
    }
    auto x_rank = x.size();
    if (x_rank < 1) {
      MS_LOG(EXCEPTION) << "For SegmentMean's gradient, the rank of input x should be greater or equal to one, but got "
                        << x_rank;
    }
    auto segment_ids_rank = segment_ids.size();
    return {SizeToLong(x_rank - 1 + segment_ids_rank)};
  });

REG_BPROP_BUILDER("SegmentMean").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto ones_shape = ib->ShapeCalc(g_segment_mean, {input_x, segment_ids})[0];
  auto ones = ib->Fill(1.0, ones_shape, TypeId::kNumberTypeFloat32);

  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  const int64_t max_len = 1000000;
  auto scaled_grad = ib->Div(dout, ib->Emit("SegmentSum", {ones, segment_ids}, {{"max_length", MakeValue(max_len)}}));
  auto dx = ib->Gather(scaled_grad, segment_ids, 0);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->OutZeros(segment_ids)};
});

REG_BPROP_BUILDER("MaskedScatter").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  dout = ib->Cast(dout, kFloat32);
  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    dx = ib->Emit("MaskedFill", {dout, mask, ib->Tensor(0, kFloat32)});
    dx = ib->Cast(dx, ib->GetDtype(x));
  } else {
    dx = ib->OutZeros(x);
  }
  NodePtr dupdates = nullptr;
  if (updates->need_compute_grad_out()) {
    dupdates = ib->Cast(ib->Reshape(ib->ZerosLike(updates), {-1}), kFloat32);
    auto dupdates_val = ib->Cast(ib->Emit("MaskedSelect", {dout, mask}), kFloat32);
    auto length = ib->TupleGetItem(ib->Shape(dupdates_val), LongToSize(0));
    auto scatter_indices = ib->Range(length);
    dupdates = ib->Emit("TensorScatterElements", {dupdates, scatter_indices, dupdates_val},
                        {{"reduction", MakeValue<string>("none")}, {"axis", MakeValue<int64_t>(0)}});
    // The operator test case pass on cpu or ascend backend. But it may fail once enabled on gpu backend for pynative
    // mode. Now it is not supported on gpu backend.
    dupdates = ib->Reshape(dupdates, ib->Shape(updates));
    dupdates = ib->Cast(dupdates, ib->GetDtype(updates));
  } else {
    dupdates = ib->OutZeros(updates);
  }
  return {dx, ib->OutZeros(mask), dupdates};
});

REG_BPROP_BUILDER("CountNonZero").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ParameterizedTruncatedNormal").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Ones").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto dims = ib->GetInput(kIndex0);
  auto type = ib->GetInput(kIndex1);
  return {ib->OutZeros(dims), ib->OutZeros(type)};
});

REG_BPROP_BUILDER("Zeros").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto dims = ib->GetInput(kIndex0);
  auto type = ib->GetInput(kIndex1);
  return {ib->OutZeros(dims), ib->OutZeros(type)};
});

REG_BPROP_BUILDER("Im2Col").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto kernel_size = GetValue<std::vector<int64_t>>(ib->GetAttr("ksizes"));
  auto dilation = GetValue<std::vector<int64_t>>(ib->GetAttr("dilations"));
  auto stride = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("pads"));
  std::vector<int64_t> padding = {pads[0], pads.back()};
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  NodePtr shape = nullptr;
  if (IsDynamic(x_shape)) {
    auto tensor_shape = ib->Emit("TensorShape", {x});
    // Im2Col only support 4-D input, so we hard-code [2:4] here
    shape = ib->StridedSlice(tensor_shape, ib->Value<ShapeVector>({2}), ib->Value<ShapeVector>({4}),
                             ib->Value<ShapeVector>({1}));
    shape = ib->Cast(shape, kInt32);
  } else {
    ShapeVector output_shape(x_shape.begin() + i2, x_shape.end());
    shape = ib->Tensor(output_shape, kInt32);
  }
  auto dx = ib->Emit("Col2Im", {dout, shape},
                     {{"kernel_size", MakeValue(kernel_size)},
                      {"dilation", MakeValue(dilation)},
                      {"stride", MakeValue(stride)},
                      {"padding", MakeValue(padding)}});
  return {dx};
});

REG_BPROP_BUILDER("TransShape").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("TransShape", {dout, ib->Shape(x)});
  return {dx, ib->OutZeros(shape)};
});

REG_BPROP_BUILDER("Max").SetBody(BODYFUNC(ib) {
  auto dx = MinOrMaxOpGrad(ib, ib->GetInput(kIndex0), ib->GetInput(kIndex1), ib->GetInput(kIndex2));
  return {dx};
});

REG_BPROP_BUILDER("Min").SetBody(BODYFUNC(ib) {
  auto dx = MinOrMaxOpGrad(ib, ib->GetInput(kIndex0), ib->GetInput(kIndex1), ib->GetInput(kIndex2));
  return {dx};
});

REG_BPROP_BUILDER("RepeatInterleave").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto repeats = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto output_size = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto axis_ptr = axis->BuildValue();
  auto axis_value = GetValue<int64_t>(axis_ptr);
  auto repeats_ptr = repeats->BuildValue();
  auto repeats_values = GetIntList(repeats_ptr);
  NodePtr result;
  if (repeats_values.size() == 1) {
    auto shape_out = ib->GetShape(dout);
    shape_out[axis_value] = shape_out[axis_value] / repeats_values[0];
    shape_out.insert(shape_out.begin() + axis_value, repeats_values[0]);
    auto reshape = ib->Reshape(dout, shape_out);
    result = ib->ReduceSum(reshape, {axis_value});
  } else {
    int idx = 0;
    NodePtrList to_merge;
    for (size_t i = 0; i < repeats_values.size(); i++) {
      std::map<int64_t, std::vector<int64_t>> slices;
      (void)slices.emplace(axis_value, std::vector<int64_t>{idx, idx + repeats_values[i]});
      auto dx = ib->StridedSlice(dout, slices);
      NodePtr rs = ib->ReduceSum(dx, {axis_value});
      auto rs_shape = rs->shape();
      rs_shape.insert(rs_shape.begin() + axis_value, 1);
      NodePtr rsh = ib->Reshape(rs, rs_shape);
      to_merge.push_back(rsh);
      idx += repeats_values[i];
    }
    result = ib->Concat(to_merge, axis_value);
  }
  return {result, ib->OutZeros(repeats), ib->OutZeros(axis), ib->OutZeros(output_size)};
});

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
