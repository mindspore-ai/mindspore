/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>
#include <memory>
#include <utility>
#include <set>
#include <string>
#include <functional>

#include "ops_utils/op_utils.h"
#include "primitive/array_op_name.h"
#include "primitive/array_ops.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"
#include "ir/functor.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "include/op_enum.h"

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

int64_t GetMindStorageSize(const std::vector<int64_t> &sizes, const std::vector<int64_t> &strides,
                           int64_t storage_offset) {
  int64_t storage_size = storage_offset + 1;
  auto rank = sizes.size();
  for (size_t i = 0; i < rank; ++i) {
    const auto &size_i = sizes[i];
    if (size_i == 0) {
      return storage_offset;
    }
    storage_size += (size_i - 1) * strides[i];
  }
  return storage_size;
}

/**
 * @brief To judge whether there may be memory overlap in as strided operator.
 *        Considering performing efficiency, is hard to judge overlap in view accurately.
 *
 *        1. Sort the strides and shape by stride ascending order.
 *        2. Traverse from left to right, it may be overlap if there exist a stride value less or equal then the maximum
 * index.
 *        3. It never overlap when all strides value larger then the corresponding maximum index.
 *
 * @param shapes Shapes of corresponding dims.
 * @param strides Strides of corresponding dims.
 * @return true: maybe overlap, false: never overlap.
 */
bool MaybeOverlapMemory(const std::vector<int64_t> &shapes, const std::vector<int64_t> &strides) {
  if (!shapes.empty()) {
    std::vector<int64_t> index_sort(shapes.size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(),
              [&](std::size_t i, std::size_t j) { return strides[i] < strides[j]; });
    int64_t max_index = 0;
    for (auto i : index_sort) {
      const auto &stride = strides[i];
      if (stride <= max_index) {
        return true;
      }
      max_index += stride * (shapes[i] - 1);
    }
  }
  return false;
}

inline NodePtr IsEmptySequence(Emitter *e, const NodePtr &x) {
  return e->Equal(e->Emit("sequence_len", {x}), e->Value<ShapeValueDType>(0));
}
}  // namespace

DEF_PURE_SHAPE_CALC(g_gather_drop_negative)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto is_pos = inputs.at(i1);
    auto gather_rank = inputs.at(i0).size();
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
    auto gather = inputs.at(i0);
    auto is_pos = inputs.at(i1);
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
    is_positive = ib->GreaterEqualScalar(ids, ib->ValueByType(0, ib->GetDtype(ids)));
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
  auto input_x = ib->GetInput(i0);
  auto segment_ids = ib->GetInput(i1);
  auto output = ib->GetInput(i2);
  auto dout = ib->GetInput(i3);
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
  auto x = ib->GetInput(i0);
  auto indices = ib->GetInput(i1);
  auto updates = ib->GetInput(i2);
  auto out = ib->GetInput(i3);
  auto dout = ib->GetInput(i4);
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

  auto x_shape = inputs.at(i0);
  auto indices_shape = inputs.at(i1);
  auto axis = inputs.at(i2);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(i3);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[i0];

  auto out_shape = RegenerateOutputShape(x_shape, indices_shape, axis_value, batch_dims_value);
  return {out_shape};
}

std::vector<int64_t> RegenerateOutputInferFunc(const ShapeArray &inputs, const HashSet<size_t> &invalid_indices) {
  constexpr size_t inputs_num = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x = inputs.at(i0);
  auto indices = inputs.at(i1);
  auto batch_dims = inputs.at(i3);
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

  auto x_shape = inputs.at(i0);
  auto dout_shape = inputs.at(i1);
  auto indices_shape = inputs.at(i2);
  auto axis = inputs.at(i3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(i4);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  auto perm_1 = GenerateShapeIndex(dout_shape, indices_shape, axis_value, batch_dims_value);
  auto perm_2 = GenerateInverseIndex(x_shape, axis_value, batch_dims_value);

  return {perm_1, perm_2};
}

std::vector<int64_t> PermsInferFunc(const ShapeArray &inputs, const HashSet<size_t> &invalid_indices) {
  auto x = inputs.at(i0);
  auto dout = inputs.at(i1);
  if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(dout)) {
    return {-1, -1};
  }

  return {SizeToLong(dout.size()), SizeToLong(x.size())};
}

DEF_PURE_SHAPE_CALC(g_calc_num_segment)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shp = inputs.at(i0);
    auto axis_v = inputs.at(i1)[0];
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

  auto values_shape = inputs.at(i0);
  auto indices_shape = inputs.at(i1);
  auto x_shape = inputs.at(i2);
  auto axis = inputs.at(i3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  if (axis_value < 0) {
    axis_value += SizeToLong(x_shape.size());
  }

  auto batch_dims = inputs.at(i4);
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

  auto values = inputs.at(i0);
  auto indices = inputs.at(i1);
  auto params_grad = inputs.at(i2);
  auto batch_dims = inputs.at(i4);

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
    auto x_shp = inputs.at(i0);
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
    auto x_shape = inputs.at(i0);
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
  auto x = ib->GetInput(i0);
  auto indices = ib->GetInput(i1);
  auto ori_indices = indices;  // indices may be changed latter.
  auto axis = ib->GetInput(i2);
  auto batch_dims_ptr = ib->GetInput(i3);
  auto out = ib->GetInput(i4);
  auto dout = ib->GetInput(i5);
  auto x_shp = ib->GetShape(x);
  auto out_shp = ib->GetShape(out);
  auto ind_shp = ib->GetShape(indices);

  if (out_shp.empty()) {
    dout = ib->ExpandDims(dout, -1);
  }
  int64_t axis_v = GetIntValue(axis);
  if (!IsDynamicRank(x_shp)) {
    axis_v = CheckRange(axis_v, SizeToLong(x_shp.size()));
  }
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
      auto out_shp1 = ib->ShapeCalc(g_regenerate_output, {x, indices, axis, batch_dims_tensor}, {i2, i3})[0];
      dout = ib->Reshape(dout, out_shp1);
    }
    // Calculate perm.
    auto perms = ib->ShapeCalc(g_perms, {x, dout, indices, axis, batch_dims_tensor}, {i3, i4});
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
  auto rank = input_shapes[i0].size();
  auto axis_s = LongToSize(NormalizeAxis(axis, rank));
  auto input_nums = input_shapes.size();

  std::vector<int64_t> split_sizes(input_nums);
  for (size_t i = 0; i < input_nums; ++i) {
    if (input_shapes[i].size() != rank) {
      MS_EXCEPTION(ValueError) << "For gradient of 'Concat', input shapes [" << i
                               << "] and input shapes [0] must have same rank, but got: " << input_shapes[i].size()
                               << " vs " << rank;
    }
    split_sizes[i] = input_shapes[i][axis_s];
  }
  auto dx = ib->SplitWithSize(dout, ib->Value(split_sizes), ib->Value(axis));

  auto dout_dtype = ib->GetDtypeId(dout);
  auto x = ib->GetInput(i0);
  NodePtrList res;
  for (size_t i = 0; i < input_nums; ++i) {
    auto dx_i = ib->TupleGetItem(dx, i);
    auto x_i = ib->TupleGetItem(x, i);
    auto x_i_dtype = ib->GetDtypeId(x_i);
    if (x_i_dtype != dout_dtype) {
      dx_i = ib->Cast(dx_i, x_i_dtype);
    }
    res.push_back(dx_i);
  }

  return {ib->MakeTuple(res)};
}

ShapeArray MeshgridDimsCal(const ShapeArray &input_shapes, ops::Indexing indexing_value) {
  ShapeArray res;
  auto single_rank = input_shapes.size();
  for (size_t i = 0; i < single_rank; ++i) {
    ShapeVector single_shape;
    for (size_t j = 0; j < single_rank; ++j) {
      if (i == j) {
        continue;
      }
      single_shape.push_back(j);
    }
    res.push_back(single_shape);
  }

  if (indexing_value == ops::Indexing::XY) {
    std::swap(res[i0], res[i1]);
  }
  return res;
}

NodePtrList StackBpropFunc(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto dout = ib->GetInput(i2);
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
  auto index = ib->GetInput(i0);
  auto x = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
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
  auto j_read = ib->Add((ib->Muls(j, ib->ValueByType(dim_at_axis_index, index_type))), index);
  auto j_read_reshape = ib->Reshape(j_read, {-1});
  auto i_after = ib->Muls(i, ib->ValueByType(dim_at_axis_output * dim_after_axis, index_type));
  auto read_id =
    ib->Add((ib->Add(i_after, (ib->Muls(j_read_reshape, ib->ValueByType(dim_after_axis, index_type))))), k);
  auto dout_reshape = ib->Reshape(dout, {-1});
  auto dx = ib->Gather(dout_reshape, read_id, ib->Value(0));
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
    auto x = inputs.at(i0);
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
  ShapeVector shape[i3] = {cond_shape, x_shape, y_shape};
  auto n = std::max(cond_size, std::max(x_size, y_size));
  std::vector<bool> need_shapecalc = {false, false, false};
  std::vector<std::vector<int64_t>> reduce_axis(i3);
  if (IsDynamicRank(shape[i0]) || IsDynamicRank(shape[i1]) || IsDynamicRank(shape[i2])) {
    return {{true, true, true}, reduce_axis};
  }
  for (size_t i = n; i >= 1; i--) {
    int64_t dim_value[i3] = {cond_size < i ? 1 : shape[i0][cond_size - i], x_size < i ? 1 : shape[i1][x_size - i],
                             y_size < i ? 1 : shape[i2][y_size - i]};
    const int64_t reduce_idx = SizeToLong(n - i);
    if (dim_value[i1] == dim_value[i0] && dim_value[i2] == dim_value[i0]) {
      if (dim_value[i0] == abstract::TensorShape::kShapeDimAny) {
        need_shapecalc[i0] = need_shapecalc[i1] = need_shapecalc[i2] = true;
        break;
      }
    } else {
      for (size_t j = 0; j < i3; j++) {
        if (dim_value[j] == 1) {
          (void)reduce_axis[j].emplace_back(reduce_idx);
        } else if (dim_value[j] == abstract::TensorShape::kShapeDimAny) {
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
    auto shape_cond = inputs.at(i0);
    auto shape_x = inputs.at(i1);
    auto shape_y = inputs.at(i2);
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
  ShapeVector shape[] = {ib->GetShape(inputs[i0]), ib->GetShape(inputs[i1]), ib->GetShape(inputs[i2])};
  auto [need_shapecalc, reduce_axis] = DynBroadcastGradientArgsSelect(shape[i0], shape[i1], shape[i2]);
  NodePtrList broadcast_axes;
  if (need_shapecalc[i0] || need_shapecalc[i1] || need_shapecalc[i2]) {
    broadcast_axes = ib->ShapeCalc(g_select_broadcast, {inputs[i0], inputs[i1], inputs[i2]});
  }
  for (size_t i = 1; i < i3; i++) {
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
      bool keep_dims = True;
      for (size_t j = 1; j < i3; j++) {
        if (shape[i].size() < shape[j].size()) {
          keep_dims = False;
          break;
        }
      }
      keep_dims = (!IsDynamicRank(shape[i0]) && !IsDynamicRank(shape[i1]) && !IsDynamicRank(shape[i2]) && keep_dims);
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
  if (broadcast_shape[i0].empty() || broadcast_shape[i1].empty() || broadcast_shape[i2].empty()) {
    if (broadcast_shape[index].empty()) {
      if (shift) {
        std::vector<int64_t> axis(broadcast_shape[index ^ 1].size());
        std::iota(axis.begin(), axis.end(), 0LL);
        reduce_dx = ib->SumExt(reduce_dx, ib->Value<ShapeVector>(axis), ib->Value(false));
      } else {
        reduce_dx = ib->SumExt(reduce_dx, ib->EmitValue(kNone), ib->Value(false));
      }
      return reduce_dx;
    }
  }
  if (!IsDynamic(broadcast_shape[i0]) && !IsDynamic(broadcast_shape[i1]) && !IsDynamic(broadcast_shape[i2]) &&
      shape_dynamic_dims <= 1) {
    std::vector<std::vector<int64_t>> bc_axis =
      BroadcastGradientArgsInferValueSelect(broadcast_shape[i0], broadcast_shape[i1], broadcast_shape[i2]);
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
  ShapeArray shape{ib->GetShape(inputs[i0]), ib->GetShape(inputs[i1]), ib->GetShape(inputs[i2])};
  NodePtrList reduce = {dout, dx, dy};
  if (IsDynamicRank(shape[i0]) || IsDynamicRank(shape[i1]) || IsDynamicRank(shape[i2])) {
    return DynBinopGradSelect(ib, cond, x, y, dout, dx, dy, shift);
  }
  if (shape[i0].size() <= shift && shape[i0].size() == shape[i1].size() && shape[i0].size() == shape[i2].size()) {
    return reduce;
  }
  ShapeArray broadcast_shape(i3);
  for (size_t i = 0; i < i3; i++) {
    broadcast_shape[i] = ShapeVector(shape[i].begin(), shape[i].end() - shift);
  }
  bool is_x_shape_dynamic = false;
  bool is_y_shape_dynamic = false;
  if (dx != nullptr) {
    reduce[i1] = StaticBinopGradSelect(ib, reduce[i1], shape, broadcast_shape, shift, i1, &is_x_shape_dynamic);
  }
  if (dy != nullptr) {
    reduce[i2] = StaticBinopGradSelect(ib, reduce[i2], shape, broadcast_shape, shift, i2, &is_y_shape_dynamic);
  }
  if (is_x_shape_dynamic || is_y_shape_dynamic) {
    return DynBinopGradSelect(ib, cond, x, y, dout, dx, dy, shift);
  }
  return reduce;
}

NodePtrList BroadcastToBackward(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto dout = ib->GetInput(i3);
  auto x_shape = ib->GetShape(x);
  auto dout_shape = ib->GetShape(dout);

  bool input_dynamic = IsDynamic(x_shape) || IsDynamic(dout_shape);
  if (!input_dynamic && x_shape == dout_shape) {
    return {dout, ib->OutZeros(ib->GetInput(i1))};
  }

  auto x_shape_node = ib->Shape(x);
  auto broadcast_axes = ib->BroadcastGradientArgs(dout, x);
  MS_EXCEPTION_IF_CHECK_FAIL(!broadcast_axes.empty(), "BroadcastGradientArgs out should not be empty!");
  auto reduction_axes = broadcast_axes[i1];
  NodePtr reduced_grad = dout;
  auto abs = reduction_axes->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (is_dyn_seq) {
    reduced_grad = ib->ReduceSum(dout, reduction_axes, true, true);
  } else {
    auto sequence_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    if (sequence_shape->size() != 0) {
      reduced_grad = ib->SumExt(dout, reduction_axes, ib->Value(true));
    }
  }
  auto dx = ib->Reshape(reduced_grad, x_shape_node);
  return {dx, ib->OutZeros(ib->GetInput(i1))};
}

REG_BPROP_BUILDERS_BEGIN(GradArrayOps)
REG_BPROP_BUILDER("GatherD").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->GatherDGradV2(x, dim, index, dout);
  return {dx, ib->OutZeros(dim), ib->OutZeros(index)};
});

REG_BPROP_BUILDER("GatherDGrad").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGrad");
});

REG_BPROP_BUILDER("GatherDGradV2").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGradV2");
});

REG_BPROP_BUILDER("SparseGatherV2").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto indices = ib->GetInput(i1);
  const auto &axis = ib->GetInput(i2);
  auto dout = ib->GetInput(i4);
  auto x_shp = ib->GetShape(x);
  auto axis_int = GetIntList(axis->BuildValue())[0];
  if (!IsDynamicRank(x_shp)) {
    axis_int = CheckRange(axis_int, SizeToLong(x_shp.size()));
  }
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
    auto x_size = std::accumulate(top_k_input_shape.begin(), top_k_input_shape.end(), static_cast<int64_t>(1),
                                  std::multiplies<int64_t>());
    auto outer_dim = std::accumulate(indices_shape.begin(), indices_shape.end() - 1, static_cast<int64_t>(1),
                                     std::multiplies<int64_t>());
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
  auto input_x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto res1 = ib->ShapeCalc(std::make_shared<SortShapeCalc1>(axis), {input_x});
  auto k = res1[0];
  if (k->abstract()->isa<abstract::AbstractSequence>()) {
    if (k->input_type() == InputType::kConstant) {
      auto value = GetIntList(k);
      k = ib->Tensor(value.at(i0), kInt64);
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

REG_BPROP_BUILDER("SortExt").FreeUselessValues_IO({i0, i2, i3}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);

  auto indices = ib->TupleGetItem(ib->GetInput(i4), i1);
  auto dout0 = ib->TupleGetItem(ib->GetInput(i5), i0);
  auto res = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout0))));
  (void)ib->InplaceScatterSrc(res, dim, indices, dout0);
  return {res, ib->OutZeros(dim), ib->OutZeros(ib->GetInput(i2)), ib->OutZeros(ib->GetInput(i3))};
});

REG_BPROP_BUILDER("Identity").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {dout};
});

REG_BPROP_BUILDER("InnerMoeTokenUnpermute").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &permuted_tokens = ib->GetInput(i0);
  const auto &sorted_indices = ib->GetInput(i1);
  const auto &probs = ib->GetInput(i2);
  const auto &padded_mode = ib->GetInput(i3);
  const auto &restore_shape = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  ShapeVector restore_shape_val = {1, 1};
  auto padded_mode_type = padded_mode->abstract()->BuildType();
  NodePtr res_grad;
  if (padded_mode_type->isa<TypeNone>()) {
    res_grad = ib->MoeTokenUnpermuteGrad(permuted_tokens, dout, sorted_indices, probs, ib->Value<bool>(false),
                                         ib->EmitValue(MakeValue(restore_shape_val)));
  } else {
    res_grad = ib->MoeTokenUnpermuteGrad(permuted_tokens, dout, sorted_indices, probs, padded_mode,
                                         ib->EmitValue(MakeValue(restore_shape_val)));
  }

  auto dpermuted_tokens =
    permuted_tokens->need_compute_grad_out() ? ib->TupleGetItem(res_grad, i0) : ib->OutZeros(permuted_tokens);
  auto dprobs = probs->need_compute_grad_out() ? ib->TupleGetItem(res_grad, i1) : ib->OutZeros(probs);
  return {dpermuted_tokens, ib->OutZeros(sorted_indices), dprobs, ib->OutZeros(padded_mode),
          ib->OutZeros(restore_shape)};
});

REG_BPROP_BUILDER("Clone").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {dout};
});

REG_BPROP_BUILDER("Range").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("Arange").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Pack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);
REG_BPROP_BUILDER("Stack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);

REG_BPROP_BUILDER("ReverseV2").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &axis = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->ReverseV2(dout, axis);
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Unstack").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Stack(dout, ib->GetAttr("axis"));
  return {dx};
});

REG_BPROP_BUILDER("UnstackExtView").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->StackExt(dout, dim);
  return {dx, ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("StackExt").FreeUselessValues_IO({i0, i1}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shapes = ib->GetShapes(x);
  auto x_size = x_shapes.size();
  const auto &dout = ib->GetInput(i3);
  const auto &axis_node = ib->GetInput(i1);
  auto input_shape = ib->GetShape(dout);
  if (input_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', 'x' can not be empty";
  }
  if (IsDynamicRank(input_shape)) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', DynamicRank is not supported";
  }
  auto axis_res = GetScalarValue<int64_t>(axis_node->BuildValue());
  if (!axis_res.has_value()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Stack', 'dim' can not be empty";
  }
  auto axis = axis_res.value();
  if (axis < 0) {
    axis += SizeToLong(input_shape.size());
  }
  auto ret = ib->UnstackExtView(dout, ib->Value(axis));
  NodePtrList res;
  for (size_t i = 0; i < x_size; ++i) {
    auto input = ib->TupleGetItem(x, i);
    auto input_dtype = ib->GetDtype(input);
    res.push_back(ib->Cast(ib->TupleGetItem(ret, i), input_dtype));
  }
  return {ib->MakeTuple(res), ib->OutZeros(axis_node)};
});

REG_BPROP_BUILDER("Contiguous").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  // Transparent dout.
  return {ib->GetInput(i2)};
});

REG_BPROP_BUILDER("StridedSlice").SetUnusedInputs({i0, i9}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &begin = ib->GetInput(i1);
  const auto &end = ib->GetInput(i2);
  const auto &strides = ib->GetInput(i3);
  const auto &begin_mask = ib->GetInput(i4);
  const auto &end_mask = ib->GetInput(i5);
  const auto &ellipsis_mask = ib->GetInput(i6);
  const auto &new_axis_mask = ib->GetInput(i7);
  const auto &shrink_axis_mask = ib->GetInput(i8);
  const auto &dout = ib->GetInput(i10);
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
  const auto &shapex = ib->GetInput(i1);
  const auto &begin = ib->GetInput(i2);
  const auto &end = ib->GetInput(i3);
  const auto &strides = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
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
  const auto &n = ib->GetInput(i0);
  const auto &m = ib->GetInput(i1);
  const auto &t = ib->GetInput(i2);
  return {ib->OutZeros(n), ib->OutZeros(m), ib->OutZeros(t)};
});

REG_BPROP_BUILDER("Select").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &cond = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &y = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? ib->Select(cond, dout, ib->ZerosLikeExt(dout, ib->EmitValue(kNone))) : nullptr;
  auto dy = y->need_compute_grad_out() ? ib->Select(cond, ib->ZerosLikeExt(dout, ib->EmitValue(kNone)), dout) : nullptr;
  auto ret = BinopGradSelect(ib, cond, x, y, dout, dx, dy);
  return {ib->OutZeros(cond), ret[i1], ret[i2]};
});

REG_BPROP_BUILDER("EmptyLike").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Empty").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NewEmpty").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("FullLike").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("OnesLike").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ZerosLike").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("OnesLikeExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ZerosLikeExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NewOnes").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NewZeros").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NewFull").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

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
    auto x = inputs.at(i0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    auto rank = SizeToLong(x.size());
    return {rank > 2 ? (rank - 2) : 0};
  });
REG_BPROP_BUILDER("ResizeNearestNeighbor").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &align_corners = ib->GetInput(i2);
  const auto &half_pixel_centers = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
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
  return {dx, ib->OutZeros(ib->GetInput(i1)), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("GatherNd").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shp = ib->Shape(x);
  return {ib->ScatterNd(ib->Cast(indices, kInt64), dout, shp), ib->OutZeros(indices)};
});

REG_BPROP_BUILDER("ScatterNd").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i0);
  const auto &shape = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  return {ib->OutZeros(indices), ib->GatherNd(dout, indices), ib->OutZeros(shape)};
});

REG_BPROP_BUILDER("Scatter").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &src = ib->GetInput(i3);
  const auto &reduce = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto input_grad = input->need_compute_grad_out()
                      ? ib->ScatterValue(dout, dim, index, ib->EmitValue(MakeValue<float>(0)), reduce)
                      : ib->OutZeros(input);
  auto idx_shape = ib->GetShape(index);
  if (IsShapeNone(idx_shape) || !src->need_compute_grad_out()) {
    return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), ib->OutZeros(src), ib->OutZeros(reduce)};
  }
  auto src_grad = ib->GatherD(dout, dim, index);
  return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), src_grad, ib->OutZeros(reduce)};
});

REG_BPROP_BUILDER("InplaceScatterSrc").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &src = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto input_grad = input->need_compute_grad_out()
                      ? (IsShapeNone(ib->GetShape(dout))  // ScatterValue does not accepts empty input
                           ? dout
                           : ib->ScatterValue(dout, dim, index, ib->EmitValue(MakeValue<float>(0)),
                                              ib->EmitValue(MakeValue<int64_t>(0))))
                      : ib->OutZeros(input);
  auto src_grad = src->need_compute_grad_out() && !IsShapeNone(ib->GetShape(index)) ? ib->GatherD(dout, dim, index)
                                                                                    : ib->OutZeros(src);
  return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), src_grad};
});

// Grad of inplace scatter with reduce is not supported, returns zeros of self / dim / index / src / reduce
REG_BPROP_BUILDER("InplaceScatterSrcReduce").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ScatterValue").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &src = ib->GetInput(i3);
  const auto &reduce = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto input_grad = ib->ScatterValue(dout, dim, index, ib->EmitValue(MakeValue<float>(0)), reduce);
  return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), ib->OutZeros(src), ib->OutZeros(reduce)};
});

REG_BPROP_BUILDER("InplaceScatterValue").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto input_grad = input->need_compute_grad_out()
                      ? (IsShapeNone(ib->GetShape(dout))  // ScatterValue does not accepts empty input
                           ? dout
                           : ib->ScatterValue(dout, dim, index, ib->EmitValue(MakeValue<float>(0)),
                                              ib->EmitValue(MakeValue<int64_t>(0))))
                      : ib->OutZeros(input);
  return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), ib->OutZeros(value)};
});

// Grad of inplace scatter with reduce is not supported, returns zeros of self / dim / index / value / reduce
REG_BPROP_BUILDER("InplaceScatterValueReduce").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ScatterNdUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = updates->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("ScatterNonAliasingAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("TensorScatterUpdate").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto x_grad =
    x->need_compute_grad_out() ? ib->TensorScatterUpdate(dout, indices, ib->ZerosLike(update)) : ib->OutZeros(x);
  auto updates_grad = update->need_compute_grad_out() ? ib->GatherNd(dout, indices) : ib->OutZeros(x);
  return {x_grad, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("Flatten").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->GetShape(x);
  if (IsDynamic(x_shape)) {
    return {ib->Reshape(dout, ib->Shape(x))};
  }
  return {ib->Reshape(dout, x_shape)};
});

REG_BPROP_BUILDER("FlattenExt").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &start = ib->GetInput(i1);
  const auto &end = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto x_shape = ib->Shape(x);
  return {ib->Reshape(dout, x_shape), ib->OutZeros(start), ib->OutZeros(end)};
});

REG_BPROP_BUILDER("Reshape").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &shp = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->Reshape(dout, ib->Shape(x));
  return {dx, ib->OutZeros(shp)};
});

REG_BPROP_BUILDER("ViewAs").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shape_x = ib->Shape(x);
  NodePtr dx;
  dx = ib->Reshape(dout, shape_x);
  return {dx, ib->OutZeros(other)};
});

REG_BPROP_BUILDER("NonZero").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NonZeroExt").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Argmax").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ArgMaxExt").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Argmin").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ArgMinExt").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ArgSort").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("IsNegInf").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Diag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {ib->Emit("DiagPart", {dout})};
});

REG_BPROP_BUILDER("DiagPart").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {ib->Emit("Diag", {dout})};
});

REG_BPROP_BUILDER("SpaceToBatch").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto dx =
    ib->Emit("BatchToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto dx =
    ib->Emit("SpaceToBatch", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("ReverseSequence").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &seq_lengths = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("ReverseSequence", {dout, seq_lengths},
                     {{"batch_dim", ib->GetAttr("batch_dim")}, {"seq_dim", ib->GetAttr("seq_dim")}});
  return {dx, ib->OutZeros(seq_lengths)};
});

REG_BPROP_BUILDER("TensorScatterAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->Emit("GatherNdExt", {dout, indices}) : ib->OutZeros(update);
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
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i3);

  const auto &axis_node = ib->GetInput(i1);

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
      auto axis_res = GetScalarValue<int64_t>(axis_node->BuildValue());
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
      auto input = ib->TupleGetItem(x, i);
      auto slice_out = ib->Slice(dout, concat_offset[i], ib->Shape(input));
      if (ib->GetDtypeId(input) != ib->GetDtypeId(slice_out)) {
        slice_out = ib->Cast(slice_out, ib->GetDtypeId(input));
      }
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
    auto new_i = e->Add(i, e->Tensor(1, kInt64));
    return {x, dout, concat_offset, new_i, new_list};
  };
  auto cond = ib->Less(i, len);
  auto while_block = ib->While(cond, while_body, {x, dout, concat_offset, i, res_list});
  // The `res_list` is a list, it should return a Tuple type rigorously.
  // Because there are no ListToTuple, and list type is ok for now.....
  auto dyn_list = ib->TupleGetItem(while_block, i4);
  return {dyn_list, ib->OutZeros(axis_node)};
});

DEF_PURE_SHAPE_CALC(g_meshgrid)
  .SetCalc([](const ShapeArray &inputs, const ElemPosIdx &pos_idx) -> ShapeArray {
    auto indexing = inputs[pos_idx[i1].front()][i0];
    auto indexing_value = static_cast<ops::Indexing>(indexing);
    ShapeArray input_shapes;
    input_shapes.reserve(pos_idx[i0].size());
    for (auto idx : pos_idx[i0]) {
      input_shapes.push_back(inputs[idx]);
    }
    return MeshgridDimsCal(input_shapes, indexing_value);
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

REG_BPROP_BUILDER("Meshgrid").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indexing_node = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  auto x_abs = x->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  auto base_shape = x_abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto x_is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (!x_is_dyn_seq) {
    auto input_shapes = ib->GetShapes(x);
    if (input_shapes.empty()) {
      MS_EXCEPTION(ValueError) << "For gradient of 'Meshgrid', 'input' can not be empty";
    }
    if (!std::any_of(input_shapes.cbegin(), input_shapes.cend(),
                     [](const std::vector<int64_t> &shape) { return IsDynamic(shape); })) {
      auto indexing_opt = GetScalarValue<int64_t>(indexing_node->BuildValue());
      if (indexing_opt.has_value()) {
        auto indexing_value = indexing_opt.value();
        auto reduce_dims = MeshgridDimsCal(input_shapes, static_cast<ops::Indexing>(indexing_value));
        NodePtrList res;
        for (size_t i = 0; i < input_shapes.size(); ++i) {
          auto reduce_value = ib->Value(reduce_dims[i]);
          auto single_dout = ib->TupleGetItem(dout, i);
          auto sum_out = ib->SumExt(single_dout, reduce_value, ib->Value(false));
          auto reshape_out = ib->Reshape(sum_out, ib->Value(input_shapes[i]));
          res.push_back(reshape_out);
        }
        return {ib->MakeTuple(res), ib->OutZeros(indexing_node)};
      }
    }

    auto reduce_dims = ib->ShapeCalc(g_meshgrid, {x, indexing_node}, {1});
    auto input_nums = input_shapes.size();
    if (reduce_dims.size() != input_nums) {
      MS_LOG(EXCEPTION) << "The number of Meshgrid's ShapeCalc(" << reduce_dims.size() << ") is not equal to input("
                        << input_nums << ")!";
    }

    NodePtrList tuple_out;
    for (size_t i = 0; i < input_nums; ++i) {
      auto single_dout = ib->TupleGetItem(dout, i);
      auto sum_out = ib->SumExt(single_dout, reduce_dims[i], ib->Value(false));
      auto x_shape = ib->Shape(ib->TupleGetItem(x, i));
      auto reshape_out = ib->Reshape(sum_out, x_shape);
      tuple_out.push_back(reshape_out);
    }
    auto res = ib->MakeTuple(tuple_out);
    return {res, ib->OutZeros(indexing_node)};
  }

  // Here the x is a dynamic sequence, so the infer out is a dynamic sequence too!
  auto reduce_dims = ib->ShapeCalc(g_meshgrid, {x, indexing_node}, {1})[i0];
  auto first_dout = ib->Shape(ib->TupleGetItem(dout, 0));
  auto reduce_dim = ib->TupleGetItem(reduce_dims, 0);
  auto first_sum_out = ib->SumExt(first_dout, reduce_dim, ib->Value(false));
  auto first_x = ib->TupleGetItem(x, 0);
  auto first_x_shape = ib->Shape(first_x);
  auto first_reshape_out = ib->Reshape(first_sum_out, first_x_shape);
  auto res_list = ib->MakeList({first_reshape_out});

  // Cannot use `auto i = ib->Tensor(1, kInt64);`.
  // If so, the i will be treat as a const Tensor but variable expected which will be not caught as a while body
  // parameter.
  // Here Emit a `ScalarToTensor` to make it a fake-variable-in-logit node.
  auto i = ib->Emit("ScalarToTensor", {ib->Value<int64_t>(1), ib->Value<int64_t>(kInt64->type_id())});
  auto len = ib->Emit("ScalarToTensor", {ib->Emit("sequence_len", {x}), ib->Value<int64_t>(kInt64->type_id())});
  auto while_body = [&x, &dout, &reduce_dims, &i, &res_list](Emitter *e) -> NodePtrList {
    auto scalar_i = e->Emit("TensorToScalar", {i});
    auto temp_dout = e->Shape(e->Emit(kTupleGetItemOpName, {dout, scalar_i}));
    auto single_dims = e->Emit(kTupleGetItemOpName, {reduce_dims, scalar_i});
    auto sum_out = e->SumExt(temp_dout, single_dims, e->Value(false));
    auto x_shape = e->Shape(e->Emit(kTupleGetItemOpName, {x, scalar_i}));
    auto reshape_out = e->Reshape(sum_out, x_shape);
    auto new_list = e->Emit("ListAppend", {res_list, reshape_out});
    auto new_i = e->Add(i, e->Tensor(1, kInt64));
    return {x, dout, reduce_dims, new_i, new_list};
  };
  auto cond = ib->Less(i, len);
  auto while_block = ib->While(cond, while_body, {x, dout, reduce_dims, i, res_list});
  // The `res_list` is a list, it should return a Tuple type rigorously.
  // Because there are no ListToTuple, and list type is ok for now.....
  auto dyn_list = ib->TupleGetItem(while_block, i4);
  return {dyn_list, ib->OutZeros(indexing_node)};
});

REG_BPROP_BUILDER("Mvlgamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MvlgammaGrad", {dout, x}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("TensorScatterDiv").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto in_grad = x->need_compute_grad_out() ? ib->Emit("TensorScatterDiv", {dout, indices, update}) : ib->OutZeros(x);
  NodePtr update_grad = nullptr;
  if (update->need_compute_grad_out()) {
    auto gather_update = ib->GatherNd(dout, indices);
    auto gather_x = ib->GatherNd(x, indices);
    auto mul_result = ib->Mul(update, update);
    auto neg_result = ib->Neg(mul_result);
    update_grad = ib->Mul(gather_update, (ib->Div(gather_x, neg_result)));
  } else {
    update_grad = ib->OutZeros(update);
  }
  return {in_grad, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterSub").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->Neg(ib->GatherNd(dout, indices)) : ib->OutZeros(update);
  return {dx, ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterMul").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
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
  const auto &dim = ib->GetInput(i1);
  const auto &indices = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto zero_value = ib->ZerosLike(value);
  auto x_grad = ib->Emit("IndexFill", {dout, dim, indices, zero_value});
  return {x_grad, ib->OutZeros(dim), ib->OutZeros(indices), zero_value};
});

REG_BPROP_BUILDER("Index").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto zero_x = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  // Indices is tuple[tensor]
  std::vector<ShapeVector> indices_shapes = indices->shapes();
  auto indices_nums = indices_shapes.size();
  NodePtrList indices_res;
  for (size_t i = 0; i < indices_nums; ++i) {
    indices_res.push_back(ib->OutZeros(ib->TupleGetItem(indices, i)));
  }
  auto dy = ib->InplaceIndexPut(zero_x, indices, dout, ib->Value(true));
  return {dy, ib->MakeTuple(indices_res)};
});

REG_BPROP_BUILDER("IndexFillScalar").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(i1);
  const auto &indices = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto zero_value = ib->ZerosLike(value);
  auto x_grad = ib->IndexFillScalar(dout, dim, indices, zero_value);
  return {x_grad, ib->OutZeros(dim), ib->OutZeros(indices), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("IndexFillTensor").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &indices = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto zero_value = ib->ZerosLike(value);
  auto false_value = ib->Value(false);

  NodePtr value_grad = nullptr;
  if (value->need_compute_grad_out()) {
    auto index_unsorted = ib->Unique2(indices, false_value, false_value, false_value);
    auto index_unsorted_first = ib->TupleGetItem(index_unsorted, i0);
    auto index_select_answer = ib->IndexSelect(dout, dim, index_unsorted_first);
    value_grad = ib->SumExt(index_select_answer, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone));
  } else {
    value_grad = ib->OutZeros(value);
  }
  auto x_grad =
    x->need_compute_grad_out() ? ib->IndexFillScalar(dout, dim, indices, ib->Value<int64_t>(0)) : ib->OutZeros(x);
  return {x_grad, ib->OutZeros(dim), ib->OutZeros(indices), value_grad};
});

REG_BPROP_BUILDER("InplaceIndexFillScalar").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(kIndex1);
  const auto &index = ib->GetInput(kIndex2);
  const auto &value = ib->GetInput(kIndex3);
  const auto &dout = ib->GetInput(kIndex5);
  auto x_grad = ib->IndexFillScalar(dout, dim, index, ib->Value<int64_t>(0));
  return {x_grad, ib->OutZeros(dim), ib->OutZeros(index), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("InplaceIndexFillTensor").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(kIndex0);
  const auto &dim = ib->GetInput(kIndex1);
  const auto &index = ib->GetInput(kIndex2);
  const auto &value = ib->GetInput(kIndex3);
  const auto &dout = ib->GetInput(kIndex5);

  NodePtr x_grad = nullptr;
  NodePtr value_grad = nullptr;
  if (x->need_compute_grad_out()) {
    x_grad = ib->IndexFillScalar(dout, dim, index, ib->Value<int64_t>(0));
  } else {
    x_grad = ib->OutZeros(x);
  }
  if (value->need_compute_grad_out()) {
    auto index_unsorted = ib->InnerUnique(index, ib->Value(false), ib->Value(false));
    auto index_unsorted_first = ib->TupleGetItem(index_unsorted, kIndex0);
    auto index_select_answer = ib->IndexSelect(dout, dim, index_unsorted_first);
    value_grad = ib->SumExt(index_select_answer, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone));
    value_grad = ib->Cast(value_grad, ib->GetDtype(value));
  } else {
    value_grad = ib->OutZeros(value);
  }

  return {x_grad, ib->OutZeros(dim), ib->OutZeros(index), value_grad};
});

REG_BPROP_BUILDER("InplaceFillScalar").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i3);
  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, ib->OutZeros(ib->GetInput(i1))};
});

REG_BPROP_BUILDER("InplaceFillTensor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  NodePtr value_grad = nullptr;
  const auto &value = ib->GetInput(i1);
  if (value->need_compute_grad_out()) {
    value_grad = ib->SumExt(ib->GetInput(i3), ib->EmitValue(kNone), ib->Value(false));
  } else {
    value_grad = ib->OutZeros(value);
  }
  const auto &x = ib->GetInput(i3);
  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, value_grad};
});

REG_BPROP_BUILDER("InplaceMaskedFillScalar").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &mask = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  auto dout_clone = ib->Clone(dout);
  auto input_grad = ib->InplaceMaskedFillScalar(dout_clone, mask, ib->Value<float>(0));
  return {input_grad, ib->OutZeros(ib->GetInput(i1)), ib->OutZeros(ib->GetInput(i2))};
});

REG_BPROP_BUILDER("MaskedFillScalar").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &mask = ib->GetInput(kIndex1);
  const auto &value = ib->GetInput(kIndex2);
  const auto &dout = ib->GetInput(kIndex4);
  auto input_grad = ib->MaskedFillScalar(dout, mask, ib->Value<float>(0));
  return {input_grad, ib->OutZeros(mask), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("InplaceMaskedFillTensor").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &mask = ib->GetInput(i1);
  const auto &value = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr input_grad = nullptr;
  NodePtr value_grad = nullptr;
  if (input->need_compute_grad_out()) {
    auto dout_clone = ib->Clone(dout);
    input_grad = ib->InplaceMaskedFillScalar(dout_clone, mask, ib->Value<float>(0));
  } else {
    input_grad = ib->OutZeros(input);
  }
  if (value->need_compute_grad_out()) {
    auto temp = ib->MaskedSelect(dout, mask);
    value_grad = ib->SumExt(temp, ib->EmitValue(kNone), ib->Value(false));
  } else {
    value_grad = ib->OutZeros(value);
  }
  return {input_grad, ib->OutZeros(ib->GetInput(i1)), value_grad};
});

REG_BPROP_BUILDER("InplaceIndexCopy").SetUnusedInputs({i0, i1, i3, i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &tensor = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  NodePtr input_grad = nullptr;
  NodePtr tensor_grad = nullptr;

  if (input->need_compute_grad_out()) {
    input_grad = ib->IndexFillScalar(dout, dim, index, ib->Value<int64_t>(0));
  } else {
    input_grad = ib->OutZeros(input);
  }

  if (tensor->need_compute_grad_out()) {
    auto tensor_shape = ib->GetShape(tensor);
    if (MS_UNLIKELY(IsDynamic(tensor_shape))) {
      auto normal_tensor_branch = [&](Emitter *e) -> NodePtrList {
        return {ib->BroadcastToView(ib->IndexSelect(dout, dim, index), ib->Shape(tensor))};
      };
      auto scalar_tensor_branch = [&](Emitter *e) -> NodePtrList {
        return {ib->IndexSelect(dout, dim, ib->Squeeze(index, MakeValue(ShapeVector{0})))};
      };
      auto is_not_scalar_tensor = ib->Emit("scalar_gt", {ib->Emit("Rank", {tensor}), ib->Value<int64_t>(0)});
      tensor_grad = ib->Conditional(is_not_scalar_tensor, normal_tensor_branch, scalar_tensor_branch);
    } else {
      tensor_grad = tensor_shape.size() > 0
                      ? ib->BroadcastToView(ib->IndexSelect(dout, dim, index), ib->Value(tensor_shape))
                      : ib->IndexSelect(dout, dim, ib->Squeeze(index, MakeValue(ShapeVector{0})));
    }
  } else {
    tensor_grad = ib->OutZeros(tensor);
  }

  return {input_grad, ib->OutZeros(dim), ib->OutZeros(index), tensor_grad};
});

REG_BPROP_BUILDER("UnsortedSegmentSum").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &segment_ids = ib->GetInput(i1);
  const auto &num_segments = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  return {GatherDropNegatives(ib, dout, segment_ids, nullptr, nullptr)[0], ib->OutZeros(segment_ids),
          ib->OutZeros(num_segments)};
});

REG_BPROP_BUILDER("UnsortedSegmentMin").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &segment_ids = ib->GetInput(i1);
  const auto &num_segments = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentMax").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &segment_ids = ib->GetInput(i1);
  const auto &num_segments = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentProd").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &segment_ids = ib->GetInput(i1);
  const auto &num_segments = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);

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
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("BatchToSpaceND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpaceND").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("SpaceToBatchND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("BroadcastTo").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  return BroadcastToBackward(ib);
});

REG_BPROP_BUILDER("BroadcastToView").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  return BroadcastToBackward(ib);
});

REG_BPROP_BUILDER("ExpandAs").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto input_shape = ib->GetShape(input);
  const auto &dout = ib->GetInput(i3);
  auto dout_shape = ib->GetShape(dout);

  bool input_dynamic = IsDynamic(input_shape) || IsDynamic(dout_shape);
  if (!input_dynamic && input_shape == dout_shape) {
    return {dout, ib->OutZeros(ib->GetInput(i1))};
  }

  auto input_shape_node = ib->Shape(input);
  auto broadcast_axes = ib->BroadcastGradientArgs(dout, input);
  MS_EXCEPTION_IF_CHECK_FAIL(!broadcast_axes.empty(), "BroadcastGradientArgs out should not be empty!");
  auto reduction_axes = broadcast_axes[i1];
  NodePtr reduced_grad = nullptr;

  // Different from the reverse implementation of BroadcastTo,
  // the original ReduceSum not uses the aclnn operator.
  reduced_grad = ib->SumExt(dout, reduction_axes, ib->Value(true));
  auto dx = ib->Reshape(reduced_grad, input_shape_node);

  return {dx, ib->OutZeros(ib->GetInput(i1))};
});

REG_BPROP_BUILDER("SpaceToDepth").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {ib->Emit("DepthToSpace", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("DepthToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {ib->Emit("SpaceToDepth", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("ScatterMax").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("ScatterMin").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("ScatterUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto updates_grad = updates->need_compute_grad_out() ? ib->Gather(dout, indices, 0) : ib->OutZeros(updates);
  return {dx, ib->OutZeros(indices), updates_grad};
});

REG_BPROP_BUILDER("InplaceNormal").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &mean = ib->GetInput(i1);
  const auto &std = ib->GetInput(i2);
  const auto &seed = ib->GetInput(i3);
  const auto &offset = ib->GetInput(i4);
  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, ib->OutZeros(mean), ib->OutZeros(std), ib->OutZeros(seed), ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("NormalTensorTensor").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalTensorFloat").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalFloatTensor").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalFloatFloat").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("UniformExt").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceExponential").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceUniform").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &from = ib->GetInput(i1);
  const auto &to = ib->GetInput(i2);
  const auto &seed = ib->GetInput(i3);
  const auto &offset = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto dx = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, ib->OutZeros(from), ib->OutZeros(to), ib->OutZeros(seed), ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("ScatterAddExt").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &indices = ib->GetInput(i2);
  const auto &update = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  NodePtr x_grad = nullptr;
  x_grad = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->GatherD(dout, axis, indices) : ib->OutZeros(update);
  return {x_grad, ib->OutZeros(axis), ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("InplaceScatterAdd").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &indices = ib->GetInput(i2);
  const auto &update = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  NodePtr x_grad = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  auto update_grad = update->need_compute_grad_out() ? ib->GatherD(dout, axis, indices) : ib->OutZeros(update);
  return {x_grad, ib->OutZeros(axis), ib->OutZeros(indices), update_grad};
});

REG_BPROP_BUILDER("NormalizeSlice").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NormalizeDimIndex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &data = ib->GetInput(i0);
  return {ib->ZerosLike(data)};
});

REG_BPROP_BUILDER("NormalizeTupleIndex").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("GetSqueezeSliceShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &data = ib->GetInput(i0);
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

REG_BPROP_BUILDER("Cast").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &t = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
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

REG_BPROP_BUILDER("TypeAs").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->TypeAs(dout, x);
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("ExpandDims").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, ib->Shape(x));
  } else {
    dx = ib->Reshape(dout, shape_x);
  }
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("View").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &shape = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Reshape(dout, ib->Shape(input));
  return {dx, ib->OutZeros(shape)};
});

REG_BPROP_BUILDER("ViewDtype").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ExpandDimsView").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, ib->Shape(x));
  } else {
    dx = ib->Reshape(dout, shape_x);
  }
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Squeeze").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shapex = ib->GetShape(x);
  NodePtr dx;
  if (IsDynamic(shapex)) {
    dx = ib->Reshape(dout, ib->Shape(x));
  } else {
    dx = ib->Reshape(dout, shapex);
  }
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("Padding").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
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
    const auto &perm = inputs[0];
    std::vector<int64_t> new_perm;
    new_perm.reserve(perm.size());
    (void)std::transform(perm.begin(), perm.end(), std::back_inserter(new_perm),
                         [&perm](const int64_t v) { return v >= 0 ? v : v + SizeToLong(perm.size()); });
    auto res_perm = InvertPermutation(new_perm);
    return {res_perm};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(i0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    return {SizeToLong(x.size())};
  });

REG_BPROP_BUILDER("Transpose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &perm = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto res_perm = ib->ShapeCalc(g_transpose, {perm}, {0})[0];
  return {ib->Transpose(dout, res_perm), ib->OutZeros(perm)};
});

REG_BPROP_BUILDER("TransposeView").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &perm = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto res_perm = ib->ShapeCalc(g_transpose, {perm}, {0})[0];
  return {ib->TransposeView(dout, res_perm), ib->OutZeros(perm)};
});

REG_BPROP_BUILDER("TransposeExtView").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dim0 = ib->GetInput(i1);
  const auto &dim1 = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->TransposeExtView(dout, dim0, dim1);
  return {dx, ib->OutZeros(dim0), ib->OutZeros(dim1)};
});

REG_BPROP_BUILDER("TExt").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->GetShape(x);
  if (IsDynamicRank(x_shape) || x_shape.size() == 2) {
    return {ib->TExt(dout)};
  }
  return {dout};
});

REG_BPROP_BUILDER("Slice").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &begin = ib->GetInput(i1);
  const auto &size = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->Emit("SliceGrad", {dout, x, begin, size});
  return {dx, ib->OutZeros(begin), ib->OutZeros(size)};
});

REG_BPROP_BUILDER("Split").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  const auto &axis = ib->GetInput(i1);
  const auto &output_num = ib->GetInput(i2);
  auto axis_ptr = axis->BuildValue();
  MS_EXCEPTION_IF_NULL(axis_ptr);
  auto axis_value = GetValue<int64_t>(axis_ptr);
  auto dx = ib->Concat(dout, axis_value);
  return {dx, ib->OutZeros(axis), ib->OutZeros(output_num)};
});

REG_BPROP_BUILDER("SplitTensor").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  const auto &split_size = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto dx = ib->Concat(dout, dim);
  return {dx, ib->OutZeros(split_size), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("SplitTensorView").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  const auto &split_size = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto dx = ib->Concat(dout, dim);
  return {dx, ib->OutZeros(split_size), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("SplitWithSize").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  const auto &split_size = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto dx = ib->Concat(dout, dim);
  return {dx, ib->OutZeros(split_size), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("SplitWithSizeView").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  const auto &split_size = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto dx = ib->Concat(dout, dim);
  return {dx, ib->OutZeros(split_size), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("Chunk").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &chunks = ib->GetInput(i1);
  const auto &axis = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->Concat(dout, axis);
  return {dx, ib->OutZeros(chunks), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("ChunkView").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &chunks = ib->GetInput(i1);
  const auto &axis = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->Concat(dout, axis);
  return {dx, ib->OutZeros(chunks), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("SliceExt").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &begin = ib->GetInput(i2);
  const auto &end = ib->GetInput(i3);
  const auto &step = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  (void)ib->InplaceCopy(ib->SliceExt(dx, axis, begin, end, step), dout);
  return {dx, ib->OutZeros(axis), ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(step)};
});

REG_BPROP_BUILDER("SliceExtView").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &begin = ib->GetInput(i2);
  const auto &end = ib->GetInput(i3);
  const auto &step = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  (void)ib->InplaceCopy(ib->SliceExtView(dx, axis, begin, end, step), dout);
  return {dx, ib->OutZeros(axis), ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(step)};
});

DEF_PURE_SHAPE_CALC(g_narrow)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(i0);
    auto axis = inputs.at(i1);
    auto begin = inputs.at(i2);
    auto length = inputs.at(i3);

    MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
    auto axis_value = axis[0];
    MS_EXCEPTION_IF_CHECK_FAIL(begin.size() == 1, "begin should be a scalar.");
    auto begin_value = begin[0];
    MS_EXCEPTION_IF_CHECK_FAIL(length.size() == 1, "length should be a scalar.");
    auto length_value = length[0];

    axis_value = axis_value < 0 ? axis_value + x_shape.size() : axis_value;
    begin_value = begin_value < 0 ? begin_value + x_shape[axis_value] : begin_value;

    auto begin_shape = x_shape;
    begin_shape[axis_value] = begin_value;
    auto end_shape = x_shape;
    end_shape[axis_value] = end_shape[axis_value] - length_value - begin_value;

    return {begin_shape, end_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(i0);
    auto axis = inputs.at(i1);
    auto begin = inputs.at(i2);
    auto length = inputs.at(i3);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(axis) || IsDynamicRank(begin) ||
        IsDynamicRank(length)) {
      return {-1, -1};
    }
    auto size = SizeToLong(inputs.at(i0).size());
    return {size, size};
  });

REG_BPROP_BUILDER("Narrow").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &begin = ib->GetInput(i2);
  const auto &length = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto res = ib->ShapeCalc(g_narrow, {x, axis, begin, length}, {1, 2, 3});
  auto dx = ib->Concat(ib->MakeTuple({ib->Zeros(res[i0], ib->Value<int64_t>(ib->GetDtypeId(dout))), dout,
                                      ib->Zeros(res[i1], ib->Value<int64_t>(ib->GetDtypeId(dout)))}),
                       axis);
  return {dx, ib->OutZeros(axis), ib->OutZeros(begin), ib->OutZeros(length)};
});

REG_BPROP_BUILDER("NarrowView").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &begin = ib->GetInput(i2);
  const auto &length = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto res = ib->ShapeCalc(g_narrow, {x, axis, begin, length}, {1, 2, 3});
  auto dx = ib->Concat(ib->MakeTuple({ib->Zeros(res[i0], ib->Value<int64_t>(ib->GetDtypeId(dout))), dout,
                                      ib->Zeros(res[i1], ib->Value<int64_t>(ib->GetDtypeId(dout)))}),
                       axis);
  return {dx, ib->OutZeros(axis), ib->OutZeros(begin), ib->OutZeros(length)};
});

DEF_PURE_SHAPE_CALC(g_tile)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    // {x_shape, dims}
    auto r_shape = TileShape(inputs.at(i1), inputs.at(i0));
    ShapeVector axis;
    size_t axis_sz = r_shape.size() / 2;
    axis.reserve(axis_sz);
    for (int64_t i = 0; i < static_cast<int64_t>(axis_sz); ++i) {
      axis.push_back(i * 2);
    }
    return {r_shape, axis};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(i0);
    auto dims = inputs.at(i1);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(dims)) {
      return {-1, -1};
    }
    auto x_sz = static_cast<int64_t>(x.size());
    auto multiples_sz = static_cast<int64_t>(dims.size());
    auto max_sz = x_sz > multiples_sz ? x_sz : multiples_sz;
    return {2 * max_sz, max_sz};
  });

REG_BPROP_BUILDER("Tile").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &input_multiples = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto calc_res = ib->ShapeCalc(g_tile, {x, input_multiples}, {1});
  auto r_shape = calc_res[0];
  auto axis = calc_res[1];
  auto dout_reshaped = ib->Reshape(dout, r_shape);
  NodePtr dx;
  auto need_reduce = ib->NeedReduce(r_shape, axis, false);
  if (need_reduce.first) {
    dx = ib->SumExt(dout_reshaped, axis, ib->Value(false));
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
  const auto &x = ib->GetInput(i0);
  const auto &idx = ib->GetInput(i1);
  const auto &axis = ib->GetInput(i2);
  auto dout = ib->GetInput(i4);
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
  axis_v = GetIntValue(axis);
  if (!IsDynamicRank(x_shp)) {
    axis_v = CheckRange(axis_v, SizeToLong(x_shp.size()));
  }
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
      auto out_shp1 = ib->ShapeCalc(g_regenerate_output, {x, indices, axis, batch_dims_tensor}, {i2, i3})[0];
      dout = ib->Reshape(dout, out_shp1);
    }

    // Calculate perm.
    auto perms = ib->ShapeCalc(g_perms, {x, dout, indices, axis, batch_dims_tensor}, {i3, i4});
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

DEF_PURE_SHAPE_CALC(g_select_ext)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    auto axis = inputs.at(1);
    auto index = inputs.at(2);

    MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
    auto axis_value = axis[0];
    MS_EXCEPTION_IF_CHECK_FAIL(index.size() == 1, "begin should be a scalar.");
    auto index_value = index[0];

    axis_value = axis_value < 0 ? axis_value + x_shape.size() : axis_value;
    index_value = index_value < 0 ? index_value + x_shape[axis_value] : index_value;
    auto begin_shape = x_shape;
    begin_shape[axis_value] = index_value;
    auto end_shape = x_shape;
    end_shape[axis_value] = end_shape[axis_value] - (index_value + 1);

    return {begin_shape, end_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    auto axis = inputs.at(1);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(axis)) {
      return {-1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, size};
  });

REG_BPROP_BUILDER("SelectExtView").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto res = ib->ShapeCalc(g_select_ext, {x, axis, index}, {1, 2});
  auto dout_expand = ib->ExpandDimsView(dout, axis);
  auto dx =
    ib->Emit(kConcatOpName, {ib->MakeTuple({ib->Zeros(res[0], ib->Value<int64_t>(ib->GetDtypeId(dout))), dout_expand,
                                            ib->Zeros(res[1], ib->Value<int64_t>(ib->GetDtypeId(dout)))}),
                             axis});

  return {dx, ib->OutZeros(axis), ib->OutZeros(index)};
});

REG_BPROP_BUILDER("MatrixBandPart").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &lower = ib->GetInput(i1);
  const auto &upper = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto part = ib->Emit("MatrixBandPart", {dout, lower, upper});
  return {part, ib->OutZeros(lower), ib->OutZeros(upper)};
});

REG_BPROP_BUILDER("MatrixDiagV3").SetUnusedInputs({i0, i2, i3, i4, i5}).SetBody(BODYFUNC(ib) {
  const auto &k = ib->GetInput(i1);
  const auto &num_rows = ib->GetInput(i2);
  const auto &num_cols = ib->GetInput(i3);
  const auto &padding_value = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto part = ib->MatrixDiagPartV3(dout, k, ib->Tensor(0, ib->GetDtype(dout)), ib->GetAttr("align"));
  return {part, ib->OutZeros(k), ib->OutZeros(num_rows), ib->OutZeros(num_cols), ib->OutZeros(padding_value)};
});

REG_BPROP_BUILDER("MatrixDiagPartV3").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto align = ib->GetAttr("align");
  const auto &x = ib->GetInput(i0);
  const auto &k = ib->GetInput(i1);
  const auto &padding_value = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
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
  const auto &x = ib->GetInput(i0);
  const auto &diagonal = ib->GetInput(i1);
  const auto &k = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
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

REG_BPROP_BUILDER("SearchSorted").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("MaskedFill").FreeUselessValues_IO({i0, i2}, {}).SetBody(BODYFUNC(ib) {
  const auto &input_data = ib->GetInput(i0);
  auto mask = ib->GetInput(i1);
  const auto &value = ib->GetInput(i2);
  auto dout = ib->GetInput(i4);
  auto dmask = ib->OutZeros(mask);
  mask = ib->Cast(mask, kFloat32);
  dout = ib->Cast(dout, kFloat32);
  NodePtr dinput =
    input_data->need_compute_grad_out() ? ib->Mul(dout, ib->Sub((ib->Tensor(1, ib->GetDtype(mask))), mask)) : nullptr;
  NodePtr dvalue = value->need_compute_grad_out() ? ib->Mul(dout, mask) : nullptr;
  auto bout = BinopGradCommon(ib, input_data, mask, dinput, dvalue);

  dinput = input_data->need_compute_grad_out() ? ib->Cast(bout[0], ib->GetDtype(input_data)) : ib->OutZeros(input_data);
  if (value->need_compute_grad_out()) {
    auto mask_shape = mask->shape();
    if (IsDynamicRank(mask_shape)) {
      auto mask_rank = ib->Shape(ib->Shape(mask, true), true);
      auto axis_node = ib->Range(ib->TensorToScalar(mask_rank));
      dvalue = ib->SumExt(bout[1], ib->TensorToTuple(axis_node), ib->Value(false));
    } else {
      dvalue = ib->SumExt(bout[1], ib->EmitValue(kNone), ib->Value(false));
    }
    dvalue = ib->Cast(dvalue, ib->GetDtype(value));
  } else {
    dvalue = ib->OutZeros(value);
  }
  return {dinput, dmask, dvalue};
});

REG_BPROP_BUILDER("Coalesce").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  auto d1 = ib->TupleGetItem(dout, 0);
  auto d2 = ib->TupleGetItem(dout, 1);
  auto d3 = ib->TupleGetItem(dout, 2);
  return {d1, d2, d3};
});

REG_BPROP_BUILDER("ConjugateTranspose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &perm = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto tmp_perm = GetIntList(perm);
  auto tmp_perm_sz = SizeToLong(tmp_perm.size());
  std::vector<int64_t> new_perm;
  (void)std::transform(tmp_perm.begin(), tmp_perm.end(), std::back_inserter(new_perm),
                       [&tmp_perm, tmp_perm_sz](const int64_t v) { return v >= 0 ? v : v + tmp_perm_sz; });
  auto res_perm = InvertPermutation(new_perm);
  return {ib->Emit("ConjugateTranspose", {dout, ib->Value<ShapeVector>(res_perm)}), ib->OutZeros(perm)};
});

REG_BPROP_BUILDER("Triu").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &diagonal = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  auto dx = ib->Triu(dout, diagonal);
  return {dx, ib->OutZeros(diagonal)};
});

REG_BPROP_BUILDER("CheckNumerics").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {ib->Emit("CheckNumerics", {dout})};
});

REG_BPROP_BUILDER("IdentityN").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {dout};
});

REG_BPROP_BUILDER("ResizeNearestNeighborV2").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &align_corners = ib->GetInput(i2);
  const auto &half_pixel_centers = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto grad_in_size = ib->ShapeCalc(std::make_shared<ResizeNearestNeighborV2ShapeCalc>(true), {x})[0];
  if (grad_in_size->input_type() == InputType::kConstant) {
    grad_in_size = ib->Value<ShapeVector>(GetIntList(grad_in_size));
  }
  auto dx = ib->Emit("ResizeNearestNeighborV2Grad", {dout, grad_in_size, align_corners, half_pixel_centers});
  return {dx, ib->OutZeros(grad_in_size), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("Tril").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("Tril", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("TrilExt").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &diagonal = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->TrilExt(dout, diagonal);
  return {dx, ib->OutZeros(diagonal)};
});

REG_BPROP_BUILDER("SegmentSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &segment_ids = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
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
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &offset = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto x_shp = ib->GetShape(x);
  auto offset_v = GetIntValue(offset);
  auto new_indices = ib->SubScalar(indices, ib->ValueByType(offset_v, ib->GetDtype(indices)), ib->Value(1));
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
  return {actual_dout, ib->OutZeros(indices), ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("MaskedSelect").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &mask = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->MaskedSelectGrad(x, mask, dout);
  auto dmask = ib->Zeros(ib->Shape(mask), ib->Value(static_cast<int64_t>(ib->GetDtypeId(mask))));
  auto x_shape = ib->GetShape(x);
  auto dx_shape = ib->GetShape(dx);
  if (x_shape != dx_shape) {
    return BinopGradCommon(ib, x, mask, dx, dmask);
  } else {
    return {dx, dmask};
  }
});

REG_BPROP_BUILDER("SplitV").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto split_dim = GetValue<int64_t>(ib->GetAttr("split_dim"));
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Concat(dout, split_dim);
  return {dx};
});

REG_BPROP_BUILDER("Col2Im").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto ksizes = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilation"));
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("stride"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("padding"));
  const auto &output_size = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("Im2Col", {dout},
                     {{"ksizes", MakeValue(ksizes)},
                      {"dilations", MakeValue(dilations)},
                      {"strides", MakeValue(strides)},
                      {"padding_mode", MakeValue("CALCULATED")},
                      {"pads", MakeValue(pads)}});
  return {dx, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("Col2ImExt").SetUnusedInputs({i0, i1, i6}).SetBody(BODYFUNC(ib) {
  const auto &output_size = ib->GetInput(i1);
  const auto &kernel_size = ib->GetInput(i2);
  const auto &dilation = ib->GetInput(i3);
  const auto &padding = ib->GetInput(i4);
  const auto &stride = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i7);

  auto dx = ib->Col2ImGrad(dout, kernel_size, dilation, padding, stride);
  return {dx,
          ib->OutZeros(output_size),
          ib->OutZeros(kernel_size),
          ib->OutZeros(dilation),
          ib->OutZeros(padding),
          ib->OutZeros(stride)};
});

DEF_PURE_SHAPE_CALC(g_im2col_ext)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(i0);
    std::vector<int64_t> output_size{input_shape.begin() + input_shape.size() - 2, input_shape.end()};
    return {output_size};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    return {2};
  });

REG_BPROP_BUILDER("Im2ColExt").SetUnusedInputs({i0, i1, i6}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &dilation = ib->GetInput(i2);
  const auto &padding = ib->GetInput(i3);
  const auto &stride = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto output_size = ib->ShapeCalc(g_im2col_ext, {input})[0];
  auto dx = ib->Col2ImExt(dout, output_size, kernel_size, dilation, padding, stride);
  return {dx, ib->OutZeros(kernel_size), ib->OutZeros(dilation), ib->OutZeros(padding), ib->OutZeros(stride)};
});

REG_BPROP_BUILDER("Col2ImGrad").SetUnusedInputs({i0, i1, i6}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &dilation = ib->GetInput(i2);
  const auto &padding = ib->GetInput(i3);
  const auto &stride = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto output_size = ib->ShapeCalc(g_im2col_ext, {input})[0];
  auto dx = ib->Col2ImExt(dout, output_size, kernel_size, dilation, padding, stride);
  return {dx, ib->OutZeros(kernel_size), ib->OutZeros(dilation), ib->OutZeros(padding), ib->OutZeros(stride)};
});

REG_BPROP_BUILDER("ExtractVolumePatches").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto ksize = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto ksize_d = ksize.at(i2);
  auto ksize_h = ksize.at(i3);
  auto ksize_w = ksize.at(i4);
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->GetShape(x);
  auto x_n = x_shape.at(i0);
  auto x_c = x_shape.at(i1);
  auto x_d = x_shape.at(i2);
  auto x_h = x_shape.at(i3);
  auto x_w = x_shape.at(i4);
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
  auto out_d = out_shape.at(i2);
  auto out_h = out_shape.at(i3);
  auto out_w = out_shape.at(i4);
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
    auto output_size = GetIntList(ib->GetInput(i1));
    auto dout = ib->GetInput(i3);
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
    if (output_size.size() == i5) {
      const auto n_value = output_size[i0];
      const auto d_value = output_size[i2];
      const auto h_value = output_size[i3];
      const auto w_value = output_size[i4];
      auto vecx = (w_value != 1) ? ib->LinSpace(start, stop, ib->Value(w_value)) : zero;
      auto vecy = (h_value != 1) ? ib->LinSpace(start, stop, ib->Value(h_value)) : zero;
      auto vecz = (d_value != 1) ? ib->LinSpace(start, stop, ib->Value(d_value)) : zero;
      if (!align_corners) {
        vecx = ib->Divs(ib->Muls(vecx, ib->ValueByType(w_value - 1, kFloat32)), ib->ValueByType(w_value, kFloat32));
        vecy = ib->Divs(ib->Muls(vecy, ib->ValueByType(h_value - 1, kFloat32)), ib->ValueByType(h_value, kFloat32));
        vecz = ib->Divs(ib->Muls(vecz, ib->ValueByType(d_value - 1, kFloat32)), ib->ValueByType(d_value, kFloat32));
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
    } else if (output_size.size() == i4) {
      auto x_shape = ib->GetShape(dout);
      const auto n_value = x_shape[i0];
      const auto h_value = x_shape[i1];
      const auto w_value = x_shape[i2];
      auto vecx = (w_value != 1) ? ib->LinSpace(start, stop, ib->Value(w_value)) : zero;
      auto vecy = (h_value != 1) ? ib->LinSpace(start, stop, ib->Value(h_value)) : zero;
      if (!align_corners) {
        vecx = ib->Divs(ib->Muls(vecx, ib->ValueByType(w_value - 1, kFloat32)), ib->ValueByType(w_value, kFloat32));
        vecy = ib->Divs(ib->Muls(vecy, ib->ValueByType(h_value - 1, kFloat32)), ib->ValueByType(h_value, kFloat32));
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
    auto output_size = ib->GetInput(i1);
    auto dout = ib->GetInput(i3);
    auto dx = ib->Emit("AffineGridGrad", {dout, output_size}, {{"align_corners", ib->GetAttr("align_corners")}});
    return {dx, ib->OutZeros(output_size)};
  }
});

REG_BPROP_BUILDER("SegmentMax").SetBody(SegmentMinOrMaxGrad);
REG_BPROP_BUILDER("SegmentMin").SetBody(SegmentMinOrMaxGrad);

REG_BPROP_BUILDER("TensorScatterElements").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &axis = ib->GetInput(i3);
  const auto &reduce = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  NodePtr x_grad = nullptr;
  if (x->need_compute_grad_out()) {
    x_grad = ib->Emit("TensorScatterElements", {dout, indices, ib->ZerosLike(update), axis, reduce});
  } else {
    x_grad = ib->OutZeros(x);
  }
  auto update_grad = update->need_compute_grad_out() ? ib->GatherD(dout, axis, indices) : ib->OutZeros(update);
  return {x_grad, ib->OutZeros(indices), update_grad, ib->OutZeros(axis), ib->OutZeros(reduce)};
});

REG_BPROP_BUILDER("ScatterAddWithAxis").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetAttr("axis");
  const auto &x = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &update = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
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
  const auto &dout = ib->GetInput(i3);
  return {ib->OutZeros(dout), dout};
});

REG_BPROP_BUILDER("Expand").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &shape = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
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
    auto x_rank = inputs.at(i0).size();
    auto segment_ids_shape = inputs.at(i1);
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
    auto x = inputs.at(i0);
    auto segment_ids = inputs.at(i1);
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
  auto input_x = ib->GetInput(i0);
  const auto &segment_ids = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
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

namespace {
NodePtr CalDupdates(BpropBuilder *ib, int64_t diffNumel, const NodePtr &dout, const NodePtr &maskSelected) {
  if (diffNumel > 0) {
    ShapeVector zerosShape = {static_cast<int64_t>(diffNumel)};
    auto zerosFillin = ib->Zeros(ib->Value<ShapeVector>(zerosShape), ib->Value<int64_t>(ib->GetDtypeId(dout)));
    auto dupdatesConcated = ib->Concat({maskSelected, zerosFillin}, 0);
    return dupdatesConcated;
  } else {
    return maskSelected;
  }
}
}  // namespace

DEF_PURE_SHAPE_CALC(g_masked_scatter)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto updatesSize =
      std::accumulate(inputs.at(kIndex0).begin(), inputs.at(kIndex0).end(), 1, std::multiplies<size_t>());
    auto maskSize = std::accumulate(inputs.at(kIndex1).begin(), inputs.at(kIndex1).end(), 1, std::multiplies<size_t>());
    std::vector<int64_t> res_shape;
    auto diffNumel = updatesSize - maskSize;
    res_shape.push_back(diffNumel);
    return {res_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

REG_BPROP_BUILDER("MaskedScatter").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &mask = ib->GetInput(i1);
  const auto &updates = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;
  // dx
  if (x->need_compute_grad_out()) {
    dx = ib->MaskedFill(dout, mask, ib->Tensor(0, ib->GetDtype(x)));
  } else {
    dx = ib->OutZeros(x);
  }

  // dupdates
  NodePtr dupdates = nullptr;
  if (updates->need_compute_grad_out()) {
    auto maskSelected = ib->MaskedSelect(dout, mask);
    if (IsDynamic(ib->GetShape(maskSelected))) {
      auto zerosShape = ib->ShapeCalc(g_masked_scatter, {updates, maskSelected})[0];
      auto diffNumel = ib->TupleGetItem(zerosShape, 0);
      auto cond = ib->Greater(ib->ScalarToTensor(diffNumel, kInt64), ib->Tensor(0, kInt64));
      auto zerosFillin = ib->Zeros(zerosShape, ib->Value<int64_t>(ib->GetDtypeId(dout)));
      auto trueBranch = [&](Emitter *e) -> NodePtrList { return {ib->Concat({maskSelected, zerosFillin}, 0)}; };
      auto falseBranch = [&maskSelected](Emitter *e) -> NodePtrList { return {maskSelected}; };
      dupdates = ib->Conditional(cond, trueBranch, falseBranch);
    } else {
      auto diffNumel = ib->GetSize(updates) - ib->GetSize(maskSelected);
      dupdates = CalDupdates(ib, diffNumel, dout, maskSelected);
    }
    dupdates = ib->Reshape(dupdates, ib->Shape(updates));
  } else {
    dupdates = ib->OutZeros(updates);
  }
  std::vector<NodePtr> ret = BinopGradCommon(ib, x, mask, dx, nullptr);
  return {ret[0], ib->OutZeros(mask), dupdates};
});

DEF_PURE_SHAPE_CALC(inplace_masked_scatter)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto source_shape = inputs[0];
    auto mask_selected_shape = inputs[1];
    int64_t source_numel = 1;
    int64_t mask_selected_numel = 1;
    for (auto size : source_shape) {
      source_numel *= size;
    }
    for (auto size : mask_selected_shape) {
      mask_selected_numel *= size;
    }
    return {{source_numel - mask_selected_numel}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    return {1};
  });

REG_BPROP_BUILDER("InplaceMaskedScatter").FreeUselessValues_IO({i0, i2}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(kIndex0);
  const auto &mask = ib->GetInput(kIndex1);
  const auto &source = ib->GetInput(kIndex2);
  const auto &dout = ib->GetInput(kIndex4);
  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    dx = ib->MaskedFillScalar(dout, mask, ib->Value<float>(0));
  } else {
    dx = ib->OutZeros(x);
  }
  NodePtr dsource = nullptr;
  if (source->need_compute_grad_out()) {
    auto mask_selected = ib->MaskedSelect(dout, mask);
    auto mask_selected_shape = ib->GetShape(mask_selected);
    auto source_shape = ib->GetShape(source);
    auto dout_type = ib->GetDtypeId(dout);
    if (IsDynamic(mask_selected_shape) || IsDynamic(source_shape)) {
      auto diff_nelem_tuple = ib->ShapeCalc(inplace_masked_scatter, {source, mask_selected})[0];
      auto diff_nelem = ib->TupleGetItem(diff_nelem_tuple, kIndex0);
      auto true_branch = [&](Emitter *e) -> NodePtrList {
        auto zeros_fillin = e->Zeros(diff_nelem_tuple, e->Value(static_cast<int64_t>(dout_type)));
        auto true_dsource = e->Concat(e->MakeTuple({mask_selected, zeros_fillin}), e->Value<int64_t>(0));
        return {true_dsource};
      };
      auto false_branch = [&](Emitter *e) -> NodePtrList { return {mask_selected}; };
      auto gt_true = ib->Emit("ScalarGt", {diff_nelem, ib->Value<int64_t>(0)});
      mask_selected = ib->Conditional(gt_true, true_branch, false_branch);
    } else {
      int64_t source_numel = 1;
      int64_t mask_selected_numel = 1;
      for (auto size : source_shape) {
        source_numel *= size;
      }
      for (auto size : mask_selected_shape) {
        mask_selected_numel *= size;
      }
      auto diff_nelem = source_numel - mask_selected_numel;
      if (diff_nelem > 0) {
        auto zeros_fillin = ib->Zeros(ib->Value<ShapeVector>({diff_nelem}), ib->Value(static_cast<int64_t>(dout_type)));
        mask_selected = ib->Concat(ib->MakeTuple({mask_selected, zeros_fillin}), ib->Value<int64_t>(0));
      }
    }
    dsource = ib->Reshape(mask_selected, ib->Shape(source));
  } else {
    dsource = ib->OutZeros(source);
  }
  return {dx, ib->OutZeros(mask), dsource};
});

REG_BPROP_BUILDER("CountNonZero").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ParameterizedTruncatedNormal").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Ones").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &dims = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  return {ib->OutZeros(dims), ib->OutZeros(type)};
});

REG_BPROP_BUILDER("Zeros").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &dims = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  return {ib->OutZeros(dims), ib->OutZeros(type)};
});

REG_BPROP_BUILDER("Im2Col").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto kernel_size = GetValue<std::vector<int64_t>>(ib->GetAttr("ksizes"));
  auto dilation = GetValue<std::vector<int64_t>>(ib->GetAttr("dilations"));
  auto stride = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("pads"));
  std::vector<int64_t> padding = {pads[0], pads.back()};
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
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
  const auto &x = ib->GetInput(i0);
  const auto &shape = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("TransShape", {dout, ib->Shape(x)});
  return {dx, ib->OutZeros(shape)};
});

/**
 * @brief Calculate the shape for gradient of repeat to do reduce (twice) and reshape. Cases:
 *        Generate 3 ShapeVector:
 *        0. The shape to pre-reduce the grad (if empty, skip this option);
 *        1. The new shape for the grad to call Reshape after pre-reduce;
 *        2. The dims to call ReduceSum after Reshape.
 *        If 2 is empty, the grad func should skip both step 1 and 2 (it means all of repeats are ones)
 *        except dynamic shape case in graph mode.
 *
 * @param input The shape of forward inputs[0] (input / self).
 * @param repeats The value of forward inputs[1] (repeats).
 * @return std::tuple<ShapeVector, ShapeVector, ShapeVector> Return shapes. See brief.
 */
static std::tuple<ShapeVector, ShapeVector, ShapeVector> RepeatGradShapesCalcCommonFunc(const ShapeVector &input,
                                                                                        const ShapeVector &repeats) {
  const auto input_rank = static_cast<ShapeValueDType>(input.size());
  const auto extra_rank = static_cast<ShapeValueDType>(repeats.size()) - input_rank;
  ShapeVector pre_reduce_dims{};
  pre_reduce_dims.reserve(extra_rank);
  // If has broadcasted dims, reduce them first to let the rank be smaller for reshape and reduceSum.
  for (ShapeValueDType i = 0; i < extra_rank; ++i) {
    pre_reduce_dims.push_back(i);
  }
  // For existing dims, size of each dim x of grad equals to repeats[x] * input.shape[x].
  // If repeat of this dim larger than 1,
  // reshape to split this dim into (repeats[x], input.shape[x]), and then reduce the `repeat` dim.
  // OtherWise, no need to reduce.
  ShapeVector reduce_dims{};
  ShapeVector grad_new_shape{};
  reduce_dims.reserve(input_rank);
  grad_new_shape.reserve(input.size() * i2);
  for (ShapeValueDType input_dim = 0; input_dim < input_rank; ++input_dim) {
    const auto input_size = *(input.begin() + input_dim);
    const auto repeat = *(repeats.begin() + (input_dim + extra_rank));
    if (repeat > 1) {
      reduce_dims.push_back(grad_new_shape.size());
      grad_new_shape.push_back(repeat);
    }
    grad_new_shape.push_back(repeat == 0 ? 0 : input_size);
  }
  return {pre_reduce_dims, grad_new_shape, reduce_dims};
}

// Inputs are: shape of input, value of repeats
// Generate 4 ShapeVector. If element 0 is empty, dynamic shape grad func should directly return Zeros.
// element 1 to 3 is the same as RepeatGradShapesCalcCommonFunc if returns[0] is false, else are all empty.
DEF_PURE_SHAPE_CALC(g_repeat)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    const auto &input = inputs[i0], &repeats = inputs[i1];
    if (IsShapeNone(input) || IsShapeNone(repeats)) {
      return {{}, {}, {}, {}};
    }
    const auto [pre_reduce, new_shape, reduce] = RepeatGradShapesCalcCommonFunc(input, repeats);
    return {{1}, pre_reduce, new_shape, reduce};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    const auto &input = inputs[i0], &repeats = inputs[i1];
    const auto has_unknown = IsDynamicRank(input) || IsDynamicShape(input) || !unknown_inputs.empty();
    if (has_unknown) {
      return {abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
              abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
    }
    if (IsShapeNone(input) || IsShapeNone(repeats)) {
      return {i0, i0, i0, i0};
    }
    // len(repeats) < input.rank is forbidden by forward infer
    const auto input_rank = static_cast<ShapeValueDType>(input.size());
    const auto repeats_len = static_cast<ShapeValueDType>(repeats.size());
    const auto extra_rank = repeats_len - input_rank;
    // If repeat of this dim larger than 1,
    // reshape to split this dim into (repeats[-x], input.shape[-x]), and then reduce the `repeat` dim.
    ShapeValueDType num_splited = 0;
    for (auto it = repeats.begin() + extra_rank; it != repeats.end(); ++it) {
      if (*it > 1) {
        ++num_splited;
      }
    }
    return {i1, extra_rank, input_rank + num_splited, num_splited};
  });

REG_BPROP_BUILDER("Repeat").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto input = ib->GetInput(i0);
  const auto &input_shape = ib->GetShape(input);
  const auto repeats = ib->GetInput(i1);
  const auto repeats_optional = GetArrayValue<ShapeValueDType>(repeats->BuildValue());
  auto grad = ib->GetInput(i3);
  const auto repeats_unknown = (!repeats_optional.has_value()) || repeats_optional->HasUnknownValue();
  const auto is_dyn_case = IsDynamic(input_shape) || repeats_unknown;
  if (is_dyn_case) {
    const auto calc_result = ib->ShapeCalc(g_repeat, {input, repeats}, {i1});
    const auto return_zeros = calc_result[0];  // If it's empty, directly return zeros
    const auto pre_reduce = calc_result[1], new_shape = calc_result[2], reduce = calc_result[3];
    auto input_grad = ib->Conditional(
      IsEmptySequence(ib, return_zeros),
      [&input](Emitter *e) -> NodePtrList {
        auto input_dtype = e->Value(static_cast<int64_t>(input->dtype()->type_id()));
        return {e->Zeros(e->Shape(input), input_dtype)};
      },
      [&grad, &pre_reduce, &new_shape, &reduce](Emitter *e) -> NodePtrList {
        auto grad1 = e->Conditional(
          IsEmptySequence(e, pre_reduce), [&grad](Emitter *e) -> NodePtrList { return {grad}; },
          [&grad, &pre_reduce](Emitter *e) -> NodePtrList {
            return {e->SumExt(grad, pre_reduce, e->Value<bool>(false))};
          });
        return {e->Conditional(
          IsEmptySequence(e, reduce), [&grad1](Emitter *e) -> NodePtrList { return {grad1}; },
          [&grad1, &new_shape, &reduce](Emitter *e) -> NodePtrList {
            // Slow path when need to call reduce: not all of repeats are all 1.
            return {e->SumExt(e->Reshape(grad1, new_shape), reduce, e->Value<bool>(false))};
          })};
      });
    return {input_grad, ib->OutZeros(repeats)};
  } else {
    auto repeats_val = repeats_optional->ToVector();
    // Fast path for empty
    if (IsShapeNone(input_shape) || IsShapeNone(repeats_val)) {
      auto input_dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(input)));
      auto dx = ib->Zeros(ib->Shape(input), input_dtype);
      return {dx, ib->OutZeros(repeats)};
    }
    const auto [pre_reduce, new_shape, reduce] = RepeatGradShapesCalcCommonFunc(input_shape, repeats_val);
    if (!pre_reduce.empty()) {
      // There are multiple calls of ReduceSum in benchmark, doing so to make sure the precision consistency.
      if (ops::UseOptimizedOpImpl()) {
        MS_LOG(DEBUG) << "Using optimized operator implementation in Repeat's backward.";
        grad = ib->SumExt(grad, ib->Value(pre_reduce), ib->Value<bool>(false));
      } else {
        for (size_t i = 0; i < pre_reduce.size(); ++i) {
          std::vector<int64_t> reduce_axis = {0};
          grad = ib->SumExt(grad, ib->Value(reduce_axis), ib->Value<bool>(false));
        }
      }
    }
    if (!reduce.empty()) {
      // Slow path when need to call reduce: not all of repeats are all 1.
      grad = ib->SumExt(ib->Reshape(grad, new_shape), ib->Value(reduce), ib->Value<bool>(false));
    }
    return {grad, ib->OutZeros(repeats)};
  }
});

NodePtr RepeatInterleaveIntGrad(BpropBuilder *ib, const NodePtr &dout, const NodePtr &x, const NodePtr &repeats,
                                const NodePtr &dim) {
  auto dout_shape = ib->GetShape(dout);
  auto dim_value = dim->BuildValue();
  MS_EXCEPTION_IF_NULL(dim_value);
  auto dim_pos = dim_value->isa<None>() ? ib->Value<int64_t>(-1) : dim;
  auto dim_pos_opt = GetScalarValue<int64_t>(dim_pos->BuildValue());
  auto repeats_opt = GetScalarValue<int64_t>(repeats->BuildValue());

  NodePtr dx = nullptr;
  if (MS_UNLIKELY(!dim_pos_opt.has_value() || !repeats_opt.has_value() || IsDynamic(dout_shape))) {
    // If dim/repeats/dout_shape is unknown, the fast path is invalid, because we can't get the dout_new_shape.
    auto new_repeats = ib->Reshape(ib->ScalarToTensor(repeats, kInt64), ib->Value<ShapeVector>({1}));
    auto grad = ib->RepeatInterleaveGrad(dout, new_repeats, dim_pos);
    dx = ib->Reshape(grad, ib->Shape(x));
  } else {
    // Fast gradient path of repeat_interleave with scalar repeats.
    if (MS_UNLIKELY(repeats_opt.value() == 0)) {
      return ib->OutZeros(x);
    }
    auto dout_rank = SizeToLong(dout_shape.size());
    auto real_dim = dim_pos_opt.value() < 0 ? dim_pos_opt.value() + dout_rank : dim_pos_opt.value();
    ShapeVector dout_new_shape = {};
    for (int64_t dim_idx = 0; dim_idx < dout_rank; ++dim_idx) {
      if (dim_idx != real_dim) {
        dout_new_shape.emplace_back(dout_shape[dim_idx]);
      } else {
        dout_new_shape.emplace_back(dout_shape[dim_idx] / repeats_opt.value());
        dout_new_shape.emplace_back(repeats_opt.value());
      }
    }
    auto new_view_shape = ib->Value<ShapeVector>(dout_new_shape);
    auto sum_dim = ib->Value(ShapeVector{real_dim + 1});
    auto dout_reshape = ib->Reshape(dout, new_view_shape);
    auto dout_sum = ib->SumExt(dout_reshape, sum_dim, ib->Value<bool>(false));
    dx = ib->Reshape(dout_sum, ib->Shape(x));
  }
  return dx;
}

REG_BPROP_BUILDER("RepeatInterleaveInt").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &repeats = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &output_size = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = RepeatInterleaveIntGrad(ib, dout, x, repeats, dim);
  return {dx, ib->OutZeros(repeats), ib->OutZeros(dim), ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("RepeatInterleaveTensor").FreeUselessValues_IO({i0, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &repeats = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &output_size = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto dim_value = dim->BuildValue();
  MS_EXCEPTION_IF_NULL(dim_value);
  auto real_dim = dim_value->isa<None>() ? ib->Value<int64_t>(-1) : dim;
  auto grad = ib->RepeatInterleaveGrad(dout, repeats, real_dim);
  auto dx = ib->Reshape(grad, ib->Shape(x));
  return {dx, ib->OutZeros(repeats), ib->OutZeros(dim), ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("GenerateEodMaskV2").SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  // dx
  NodePtrList gradients{ib->GetInput(i13)};
  // other gradients are all zeros
  std::transform(all_inputs.begin() + i1, all_inputs.begin() + all_inputs.size() - i2, std::back_inserter(gradients),
                 [&ib](const NodePtr &node) { return ib->OutZeros(node); });
  return gradients;
});

REG_BPROP_BUILDER("InsertGemV2InBackward").SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  // fetch dx
  auto dout = all_inputs[all_inputs.size() - i1];
  NodePtrList gem_inputs{dout};
  gem_inputs.insert(gem_inputs.end(), all_inputs.begin() + i1, all_inputs.begin() + all_inputs.size() - i2);
  auto dx = ib->Emit("GenerateEodMaskV2", gem_inputs);
  // gradients
  NodePtrList gradients{dx};
  // other gradients are all zeros
  std::transform(all_inputs.begin() + i1, all_inputs.begin() + all_inputs.size() - i2, std::back_inserter(gradients),
                 [&ib](const NodePtr &node) { return ib->OutZeros(node); });
  return gradients;
});

REG_BPROP_BUILDER("AsStrided").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &size = ib->GetInput(i1);
  const auto &strides_node = ib->GetInput(i2);
  const auto &offset = ib->GetInput(i3);
  const auto &output = ib->GetInput(i4);
  auto dout = ib->GetInput(i5);

  auto output_tensor = output->BuildValue()->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(output_tensor);
  MS_EXCEPTION_IF_NULL(output_tensor->storage_info());
  auto output_storage_offset = output_tensor->storage_info()->storage_offset;
  auto input_tensor = input->BuildValue()->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_storage_offset =
    input_tensor->storage_info() == nullptr ? 0 : input_tensor->storage_info()->storage_offset;

  auto grad_rank = output_tensor->storage_info()->shape.size();
  std::vector<int64_t> out_shape, out_strides;
  out_shape.reserve(grad_rank);
  out_strides.reserve(grad_rank);
  const auto &shape = output_tensor->storage_info()->shape;
  const auto &strides = output_tensor->storage_info()->strides;
  for (int64_t i = grad_rank - 1; i >= 0; i--) {
    int64_t axis_size = shape[i];
    int64_t stride = strides[i];
    if (axis_size == 0) {
      return {ib->OutZeros(input), ib->OutZeros(size), ib->OutZeros(strides_node), ib->OutZeros(offset)};
    } else if (axis_size == 1) {
      std::vector<int64_t> axis{i};
      dout = ib->Squeeze(dout, MakeValue(axis));
    } else if (stride == 0) {
      std::vector<int64_t> axis{i};
      dout = ib->SumExt(dout, ib->Value(axis), ib->Value(false));
    } else {
      out_shape.insert(out_shape.begin(), axis_size);
      out_strides.insert(out_strides.begin(), stride);
    }
  }
  // The result of as strided operator maybe overlap.
  auto output_maybe_overlap = MaybeOverlapMemory(out_shape, out_strides);

  auto input_rank = input_tensor->storage_info() == nullptr ? output_tensor->storage_info()->ori_shape.size()
                                                            : input_tensor->storage_info()->shape.size();
  auto input_shape = input_tensor->storage_info() == nullptr ? output_tensor->storage_info()->ori_shape
                                                             : input_tensor->storage_info()->shape;
  auto input_strides = input_tensor->storage_info() == nullptr ? output_tensor->storage_info()->ori_strides
                                                               : input_tensor->storage_info()->strides;
  std::vector<int64_t> simplify_input_shape, simplify_input_strides;
  simplify_input_shape.reserve(input_rank);
  simplify_input_strides.reserve(input_rank);
  for (int64_t i = input_rank - 1; i >= 0; i--) {
    const auto &axis_size = input_shape[i];
    const auto &stride = input_strides[i];
    if (axis_size == 0) {
      return {ib->OutZeros(input), ib->OutZeros(size), ib->OutZeros(strides_node), ib->OutZeros(offset)};
    } else if (axis_size != 1) {
      simplify_input_shape.insert(simplify_input_shape.begin(), axis_size);
      simplify_input_strides.insert(simplify_input_strides.begin(), stride);
    }
  }
  // The input of as strided maybe an overlap view.
  auto input_maybe_overlap = MaybeOverlapMemory(simplify_input_shape, simplify_input_strides);

  auto shared_offset = std::min(output_storage_offset, input_storage_offset);
  int64_t input_effective_offset = input_storage_offset - shared_offset;
  int64_t output_effective_offset = output_storage_offset - shared_offset;
  int64_t input_min_storage = GetMindStorageSize(simplify_input_shape, simplify_input_strides, input_effective_offset);
  int64_t output_min_storage = GetMindStorageSize(out_shape, out_strides, output_effective_offset);
  auto base_size = std::max(input_min_storage, output_min_storage);
  std::vector<int64_t> storage_shape{base_size};
  auto storage = ib->Zeros(ib->Value(storage_shape), ib->Value(static_cast<int64_t>(input->dtype()->type_id())));

  // Index array can be used in both input overlap and output overlap scene.
  NodePtr flatten_full_indices;
  if (output_maybe_overlap || input_maybe_overlap) {
    auto start = ib->EmitValue(MakeValue<int64_t>(0));
    auto end = ib->EmitValue(MakeValue<int64_t>(base_size));
    auto step = ib->EmitValue(MakeValue<int64_t>(1));
    auto dtype = ib->Value(static_cast<int64_t>(TypeId::kNumberTypeInt64));
    flatten_full_indices = ib->Arange(start, end, step, dtype);
  }

  // For output overlap, all change to view should be accumulated to the base tensor.
  if (output_maybe_overlap) {
    auto output_indices = ib->AsStrided(flatten_full_indices, ib->Value(out_shape), ib->Value(out_strides),
                                        ib->Value(output_effective_offset));
    storage = ib->IndexAddExt(storage, ib->Value<int64_t>(0LL), ib->Reshape(output_indices, {-1}),
                              ib->Reshape(dout, {-1}), ib->Value<int64_t>(1LL));
  } else {
    auto view_output =
      ib->AsStrided(storage, ib->Value(out_shape), ib->Value(out_strides), ib->Value(output_effective_offset));
    (void)ib->InplaceCopy(view_output, dout);
  }

  // For input overlap, all change to base tensor should be dived by count.
  if (input_maybe_overlap) {
    auto count = ib->Zeros(ib->Shape(storage), ib->Value(static_cast<int64_t>(TypeId::kNumberTypeFloat32)));
    auto input_indices = ib->AsStrided(flatten_full_indices, ib->Value(simplify_input_shape),
                                       ib->Value(simplify_input_strides), ib->Value(input_effective_offset));
    auto flatten_input_indices = ib->Reshape(input_indices, {-1});
    ShapeVector one_shape{1};
    auto source = ib->BroadcastTo(
      ib->Ones(ib->Value<ShapeVector>(one_shape), ib->Value(static_cast<int64_t>(input->dtype()->type_id()))),
      flatten_input_indices);
    count = ib->IndexAddExt(count, ib->Value<int64_t>(0LL), flatten_input_indices, source, ib->Value<int64_t>(1LL));
    storage = ib->Div(storage, count);
  }
  auto input_grad =
    ib->AsStrided(storage, ib->Value(input_shape), ib->Value(input_strides), ib->Value(input_effective_offset));
  return {input_grad, ib->OutZeros(size), ib->OutZeros(strides_node), ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("Take").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &index = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto input_zero = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  auto grad = ib->InplacePut(input_zero, index, dout, ib->Value<bool>(true));
  return {grad, ib->OutZeros(index)};
});

REG_BPROP_BUILDER("InplacePut").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &index = ib->GetInput(i1);
  const auto &source = ib->GetInput(i2);
  const auto &accumulate = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  NodePtr grad;
  auto type_node = ib->Value(static_cast<int64_t>(ib->GetDtypeId(source)));
  auto accumulate_opt = mindspore::GetScalarValue<bool>(accumulate->BuildValue());
  if (input->need_compute_grad_out()) {
    if (!accumulate_opt.has_value()) {
      auto grad_true_branch = [&](Emitter *e) -> NodePtrList {
        return {InplacePutGrad(e, index, source, true, dout, type_node)};
      };
      auto grad_false_branch = [&](Emitter *e) -> NodePtrList {
        return {InplacePutGrad(e, index, source, false, dout, type_node)};
      };
      auto accumulate_true = ib->Equal(accumulate, ib->Value<bool>(true));
      grad = ib->Conditional(accumulate_true, grad_true_branch, grad_false_branch);
    } else {
      grad = InplacePutGrad(ib, index, source, accumulate_opt.value(), dout, type_node);
    }
  } else {
    grad = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  }
  NodePtr source_grad;
  if (source->need_compute_grad_out()) {
    source_grad = ib->Reshape(ib->Take(dout, index), ib->Shape(source));
  } else {
    source_grad = ib->OutZeros(source);
  }
  auto index_grad = ib->OutZeros(index);
  return {grad, index_grad, source_grad, ib->OutZeros(accumulate)};
});

REG_BPROP_BUILDER("InplaceFillDiagonal").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &fill_value = ib->GetInput(i1);
  const auto &wrap = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto grad = ib->InplaceFillDiagonal(dout, ib->Value<float>(0), ib->Value<bool>(false));
  return {grad, ib->OutZeros(fill_value), ib->OutZeros(wrap)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
