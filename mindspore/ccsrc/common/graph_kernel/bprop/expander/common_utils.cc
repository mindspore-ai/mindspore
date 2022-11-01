/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/bprop/expander/common_utils.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <limits>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include "ops/core_ops.h"
#include "utils/anf_utils.h"

namespace mindspore::expander::bprop {
namespace {
NodePtr ReduceSumWithCast(const BpropIRBuilder *ib, const NodePtr &dx, const std::vector<int64_t> &axis) {
  auto dx_origin_dtypeptr = ib->GetDtype(dx);
  auto dx_origin_dtype = dx_origin_dtypeptr->type_id();
  if (dx_origin_dtype == TypeId::kNumberTypeInt16 || dx_origin_dtype == TypeId::kNumberTypeInt32 ||
      dx_origin_dtype == TypeId::kNumberTypeInt64) {
    auto dx_fp32 = ib->Cast(dx, kFloat32);
    auto red = ib->Emit("ReduceSum", {dx_fp32, ib->Value(axis)}, {{"keep_dims", MakeValue(false)}});
    return ib->Cast(red, dx_origin_dtypeptr);
  }
  return ib->Emit("ReduceSum", {dx, ib->Value(axis)}, {{"keep_dims", MakeValue(false)}});
}

void ComputeReduceIndex(const std::vector<int64_t> &x_rev, const std::vector<int64_t> &y_rev,
                        std::vector<int64_t> *grad_x_reduce_idx, std::vector<int64_t> *grad_y_reduce_idy) {
  MS_EXCEPTION_IF_NULL(grad_x_reduce_idx);
  MS_EXCEPTION_IF_NULL(grad_y_reduce_idy);
  const size_t n = x_rev.size();
  if (y_rev.size() < n) {
    MS_LOG(EXCEPTION) << "The size of y_rev is less than the size of x_rev.";
  }
  for (size_t i = 0; i < n; ++i) {
    const int64_t x_i = x_rev[i];
    const int64_t y_i = y_rev[i];
    const int64_t reduce_idx = SizeToLong(n - 1 - i);
    if (x_i == y_i) {
      if (x_i == 1) {
        grad_x_reduce_idx->push_back(reduce_idx);
        grad_y_reduce_idy->push_back(reduce_idx);
      }
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input for BroadcastGradientArgs.";
    }
  }

  std::reverse(grad_x_reduce_idx->begin(), grad_x_reduce_idx->end());
  std::reverse(grad_y_reduce_idy->begin(), grad_y_reduce_idy->end());
}

TypeId GetOutputDtype(TypeId t1, TypeId t2, bool use_complex = false) {
  static std::unordered_map<TypeId, int> complex_priority_map{
    {kNumberTypeFloat32, 0}, {kNumberTypeFloat32, 1}, {kNumberTypeComplex64, 2}, {kNumberTypeComplex128, 4}};
  static std::unordered_map<TypeId, int> type_priority_map{
    {kNumberTypeBool, 0},  {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},     {kNumberTypeUInt16, 3},
    {kNumberTypeInt16, 4}, {kNumberTypeUInt32, 5},  {kNumberTypeInt32, 6},    {kNumberTypeUInt64, 7},
    {kNumberTypeInt64, 8}, {kNumberTypeFloat16, 9}, {kNumberTypeFloat32, 10}, {kNumberTypeFloat64, 11}};
  int priority_1 = 0;
  int priority_2 = 0;
  if (use_complex) {
    if (complex_priority_map.find(t1) == complex_priority_map.end() ||
        complex_priority_map.find(t2) == complex_priority_map.end()) {
      MS_EXCEPTION(ValueError) << "Complex binary op type promotion not supported for " << TypeIdToString(t1) << " and "
                               << TypeIdToString(t2);
    }
    priority_1 = complex_priority_map[t1];
    priority_2 = complex_priority_map[t2];
  } else {
    if (type_priority_map.find(t1) == type_priority_map.end() ||
        type_priority_map.find(t2) == type_priority_map.end()) {
      MS_EXCEPTION(ValueError) << "Binary op type promotion not supported for " << TypeIdToString(t1) << " and "
                               << TypeIdToString(t2);
    }
    priority_1 = type_priority_map[t1];
    priority_2 = type_priority_map[t2];
  }
  return (priority_1 > priority_2 ? t1 : t2);
}
}  // namespace

ShapeVector GetAxisValue(const NodePtr &axis) {
  MS_EXCEPTION_IF_NULL(axis);
  auto axis_node = axis->get<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(axis_node);
  return GetAxisList(axis_node->value());
}

std::pair<ShapeVector, ShapeVector> SplitShapeIndex(const ShapeVector &input_shape, const ShapeVector axis) {
  auto rank = input_shape.size();
  std::set<int64_t> reduction_indices_set;
  ShapeVector perm;
  int64_t reduced_num = 1;
  int64_t other_num = 1;
  for (auto i : axis) {
    i = (i + rank) % rank;
    reduction_indices_set.insert(i);
    reduced_num *= input_shape[i];
    perm.emplace_back(i);
  }
  ShapeVector other_indices;
  for (int64_t i = 0; i < (int64_t)rank; i++) {
    if (reduction_indices_set.find(i) == reduction_indices_set.end()) {
      other_indices.emplace_back(i);
      other_num *= input_shape[i];
      perm.emplace_back(i);
    }
  }
  ShapeVector pack_shape{reduced_num, other_num};
  return std::make_pair(pack_shape, perm);
}

std::vector<std::vector<int64_t>> BroadcastGradientArgs(const std::vector<int64_t> &x_shape,
                                                        const std::vector<int64_t> &y_shape) {
  std::vector<std::vector<int64_t>> bc_axis;
  if (x_shape == y_shape) {
    (void)bc_axis.emplace_back(std::vector<int64_t>{});
    (void)bc_axis.emplace_back(std::vector<int64_t>{});
    return bc_axis;
  }
  std::vector<int64_t> reverse_x;
  std::vector<int64_t> reverse_y;

  (void)std::transform(x_shape.rbegin(), x_shape.rend(), std::back_inserter(reverse_x),
                       [](const int64_t &c) { return c; });
  (void)std::transform(y_shape.rbegin(), y_shape.rend(), std::back_inserter(reverse_y),
                       [](const int64_t &c) { return c; });

  if (reverse_x.size() > reverse_y.size()) {
    reverse_y.resize(reverse_x.size(), 1);
  } else {
    reverse_x.resize(reverse_y.size(), 1);
  }

  std::vector<int64_t> grad_x_reduce_idx;
  std::vector<int64_t> grad_y_reduce_idy;
  ComputeReduceIndex(reverse_x, reverse_y, &grad_x_reduce_idx, &grad_y_reduce_idy);

  (void)bc_axis.emplace_back(std::move(grad_x_reduce_idx));
  (void)bc_axis.emplace_back(std::move(grad_y_reduce_idy));
  return bc_axis;
}

std::vector<int64_t> TupleDiv(const std::vector<int64_t> &x, const std::vector<int64_t> &y) {
  std::vector<int64_t> out;
  if (x.size() != y.size()) {
    MS_LOG(EXCEPTION) << "The size of inputs of TupleDiv must be the same, but the size of divisor tuple is"
                      << " " << y.size() << ", the size of dividend tuple is " << x.size() << ".";
  }
  for (size_t i = 0; i < y.size(); i++) {
    if (y[i] == 0) {
      MS_LOG(EXCEPTION) << "The divisor value should not be 0!";
    }
    if ((x[i] % y[i]) != 0) {
      MS_LOG(EXCEPTION) << "The inputs of TupleDiv should be divisible, but they are not divisible now, "
                        << "the dividend is " << x[i] << ", the divisor is " << y[i] << ".";
    }
    out.push_back(x[i] / y[i]);
  }
  return out;
}

std::vector<int64_t> ReduceShape(const std::vector<int64_t> &x, const std::vector<int64_t> &axis) {
  std::vector<int64_t> out;
  if (axis.empty()) {
    return std::vector<int64_t>(x.size(), 1LL);
  }
  std::unordered_set<int64_t> axis_set;
  int64_t x_rank = SizeToLong(x.size());
  for (const auto &i : axis) {
    if (i >= x_rank || i < (-x_rank)) {
      MS_LOG(EXCEPTION) << "axis should be in range [" << (-x_rank) << ", " << x_rank << ").";
    }
    (void)axis_set.insert(i);
  }
  for (auto i = 0; i < x_rank; i++) {
    if (axis_set.count(i) > 0 || axis_set.count(i - x_rank) > 0) {
      out.push_back(1LL);
    } else {
      out.push_back(x[LongToSize(i)]);
    }
  }
  return out;
}

std::vector<int64_t> GetAxisList(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<int64_t> result;
  if (value->isa<ValueSequence>()) {
    const auto &vals = value->cast<ValueSequencePtr>()->value();
    (void)std::transform(vals.begin(), vals.end(), std::back_inserter(result),
                         [](const ValuePtr &v) { return AnfUtils::GetIntValue(v); });
  } else {
    result.push_back(AnfUtils::GetIntValue(value));
  }
  return result;
}

NodePtrList BinopGradCommon(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                            const NodePtr &dy) {
  // Common grad definition for binary operations.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  auto shape_x = ib->GetShape(x);
  auto shape_y = ib->GetShape(y);
  auto reduce_dx = dx;
  auto reduce_dy = dy;

  std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(shape_x, shape_y);
  if (!bc_axis[0].empty()) {
    auto dx_shape = ib->GetShape(dx);
    if (!dx_shape.empty()) {
      reduce_dx = ReduceSumWithCast(ib, reduce_dx, bc_axis[0]);
    }
    reduce_dx = ib->Reshape(reduce_dx, shape_x);
  }

  if (!bc_axis[1].empty()) {
    auto dy_shape = ib->GetShape(dy);
    if (!dy_shape.empty()) {
      reduce_dy = ReduceSumWithCast(ib, reduce_dy, bc_axis[1]);
    }
    reduce_dy = ib->Reshape(reduce_dy, shape_y);
  }
  return {reduce_dx, reduce_dy};
}

NodePtrList BinopGradCommonWithShift(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                                     const NodePtr &dy, size_t shift) {
  // Common grad definition for binary operations with shift.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  auto shape_x = ib->GetShape(x);
  auto shape_y = ib->GetShape(y);
  auto reduce_dx = dx;
  auto reduce_dy = dy;
  std::vector<int64_t> broadcast_shape_of_x;
  std::vector<int64_t> broadcast_shape_of_y;
  for (size_t i = 0; i < shape_x.size() - shift; i++) {
    broadcast_shape_of_x.push_back(shape_x[i]);
  }
  for (size_t i = 0; i < shape_y.size() - shift; i++) {
    broadcast_shape_of_y.push_back(shape_y[i]);
  }

  std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(broadcast_shape_of_x, broadcast_shape_of_y);
  if (!bc_axis[0].empty()) {
    auto dx_shape = ib->GetShape(dx);
    if (!dx_shape.empty()) {
      reduce_dx = ReduceSumWithCast(ib, reduce_dx, bc_axis[0]);
    }
    reduce_dx = ib->Reshape(reduce_dx, shape_x);
  }

  if (!bc_axis[1].empty()) {
    auto dy_shape = ib->GetShape(dy);
    if (!dy_shape.empty()) {
      reduce_dy = ReduceSumWithCast(ib, reduce_dy, bc_axis[1]);
    }
    reduce_dy = ib->Reshape(reduce_dy, shape_y);
  }
  return {reduce_dx, reduce_dy};
}

std::vector<int64_t> Range(int64_t start, int64_t stop, int64_t step) {
  auto size = (stop - start) / step;
  if (size <= 0) {
    return {};
  }
  size = ((stop - start) % step == 0) ? size : size + 1;
  std::vector<int64_t> range(size);
  std::generate(range.begin(), range.end(), [n = start - step, step]() mutable {
    n = n + step;
    return n;
  });
  return range;
}

std::vector<int64_t> Range(int64_t stop) { return Range(0, stop); }

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  std::vector<int64_t> reverse_axis;
  for (int64_t i = 0; i < rk; ++i) {
    reverse_axis.emplace_back(i);
  }
  reverse_axis[axis] = rk - 1;
  reverse_axis[rk - 1] = axis;
  return reverse_axis;
}

NodePtr GetEps(const BpropIRBuilder *ib, const TypePtr &type) {
  switch (type->type_id()) {
    case kNumberTypeFloat16:
      return ib->Tensor(0.000977, type);
    case kNumberTypeFloat32:
      return ib->Tensor(std::numeric_limits<float>::epsilon(), type);
    case kNumberTypeFloat64:
      return ib->Tensor(std::numeric_limits<double>::epsilon(), type);
    default:
      return ib->Tensor(0, type);
  }
}

NodePtrList BinopGatherCommon(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto orig_indices = indices;
  auto x_shp = ib->GetShape(x);
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);
  auto axis_v = GetIntFromValueNode(axis);
  if (out_shp.empty()) {
    dout = ib->Emit("ExpandDims", {dout, ib->Tensor(-1)});
  }
  if (ind_shp.empty()) {
    indices = ib->Emit("ExpandDims", {indices, ib->Tensor(-1)});
    ind_shp = ib->GetShape(indices);
    auto out_shp1 = RegenerateOutputShape(x_shp, ind_shp, axis_v);
    dout = ib->Reshape(dout, out_shp1);
  }
  out_shp = ib->GetShape(dout);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_v);
  auto values_transpose = ib->Emit("Transpose", {dout, ib->Value<ShapeVector>(perm_1)});
  auto tmp = ib->Emit("UnsortedSegmentSum", {values_transpose, indices, ib->Value<int64_t>(x_shp[axis_v])});
  auto perm_2 = GenerateInverseIndex(x_shp, axis_v);
  auto params_grad = ib->Emit("Transpose", {tmp, ib->Value<ShapeVector>(perm_2)});
  return {params_grad, ib->ZerosLike(orig_indices), ib->ZerosLike(axis)};
}

std::vector<int64_t> GenerateInverseIndex(const std::vector<int64_t> &x_shp, int64_t axis_v) {
  int64_t x_rank = static_cast<int64_t>(x_shp.size());
  auto index = Range(x_rank);
  if (axis_v < 0) {
    axis_v += x_rank;
  }
  std::vector<int64_t> perm;
  auto start1 = x_rank <= 1 ? index.end() : index.begin() + 1;
  auto end1 = axis_v + 1 >= x_rank ? index.end() : index.begin() + axis_v + 1;
  auto start2 = axis_v + 1 >= x_rank ? index.end() : index.begin() + axis_v + 1;
  (void)std::copy(start1, end1, std::back_inserter(perm));
  perm.push_back(0);
  (void)std::copy(start2, index.end(), std::back_inserter(perm));
  return perm;
}

std::vector<int64_t> GenerateShapeIndex(const std::vector<int64_t> &out_shp, const std::vector<int64_t> &ind_shp,
                                        int64_t axis_v) {
  int64_t out_rank = static_cast<int64_t>(out_shp.size());
  int64_t ind_rank = static_cast<int64_t>(ind_shp.size());
  if (axis_v < 0) {
    axis_v += out_rank - ind_rank + 1;
  }
  auto perm_part1 = Range(axis_v, axis_v + ind_rank);
  auto index = Range(out_rank);
  std::vector<int64_t> perm;
  auto end = axis_v >= out_rank ? out_rank - 1 : axis_v;
  auto start = axis_v + ind_rank >= out_rank ? index.end() : index.begin() + axis_v + ind_rank;
  (void)std::copy(perm_part1.begin(), perm_part1.end(), std::back_inserter(perm));
  (void)std::copy(index.begin(), index.begin() + end, std::back_inserter(perm));
  (void)std::copy(start, index.end(), std::back_inserter(perm));
  return perm;
}

std::vector<int64_t> RegenerateOutputShape(const std::vector<int64_t> &x_shp, const std::vector<int64_t> &ind_shp,
                                           int64_t axis_v) {
  int64_t rank = static_cast<int64_t>(x_shp.size());
  if (axis_v < 0) {
    axis_v += rank;
  }
  std::vector<int64_t> out_shp;
  auto end = axis_v >= rank ? rank - 1 : axis_v;
  auto start = axis_v + 1 >= rank ? x_shp.end() : x_shp.begin() + axis_v + 1;
  (void)std::copy(x_shp.begin(), x_shp.begin() + end, std::back_inserter(out_shp));
  (void)std::copy(ind_shp.begin(), ind_shp.end(), std::back_inserter(out_shp));
  (void)std::copy(start, x_shp.end(), std::back_inserter(out_shp));
  return out_shp;
}

std::vector<int64_t> TileShape(const std::vector<int64_t> &multiples, const std::vector<int64_t> &shapex) {
  int64_t len_multi = static_cast<int64_t>(multiples.size());
  int64_t len_shape = static_cast<int64_t>(shapex.size());
  int64_t len_cmp = len_multi - len_shape;
  auto max_len = std::max(len_multi, len_shape);
  int64_t i = 0;
  int64_t j = 0;
  std::vector<int64_t> res;
  while (i < max_len && j < max_len) {
    if (len_cmp == 0) {
      res.push_back(multiples[i]);
      res.push_back(shapex[j]);
      i++;
      j++;
    } else if (len_cmp > 0) {
      res.push_back(multiples[i]);
      res.push_back(1);
      i++;
      len_cmp--;
    } else {
      res.push_back(1);
      res.push_back(shapex[j]);
      j++;
      len_cmp++;
    }
  }
  return res;
}

std::vector<int64_t> InvertPermutation(const std::vector<int64_t> &perm) {
  std::vector<int64_t> check_perm(perm);
  std::vector<int64_t> res(perm);
  if (res.empty()) {
    return res;
  }
  std::sort(check_perm.begin(), check_perm.end());
  int64_t perm_size = static_cast<int64_t>(check_perm.size());
  for (int64_t i = 0; i < perm_size; i++) {
    if (check_perm[i] != i) {
      MS_LOG(EXCEPTION) << "For InvertPermutation, the input_x should be '[0-" << (perm_size - 1) << "]', but got "
                        << check_perm;
    }
    res[perm[i]] = i;
  }
  return res;
}

std::vector<int64_t> GetTransposition(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  auto trans = Range(axis);
  auto after_axis = Range(axis + 1, rank - 1);
  trans.push_back(rank - 1);
  trans.insert(trans.end(), after_axis.begin(), after_axis.end());
  trans.push_back(axis);
  return trans;
}

int64_t GetIntFromValueNode(const NodePtr &node) {
  if (!node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The output of node should be value node, but got " << node->get()->ToString();
  }
  auto value_node = node->get()->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto v = value_node->value();
  MS_EXCEPTION_IF_NULL(v);
  int64_t res = 0;
  if (v->isa<Int32Imm>()) {
    res = static_cast<int64_t>(GetValue<int32_t>(v));
  } else if (v->isa<Int64Imm>()) {
    res = GetValue<int64_t>(v);
  } else {
    MS_LOG(EXCEPTION) << "The output of node should be int, but got " << node->get()->ToString();
  }
  return res;
}

std::vector<int64_t> GetTupleIntFromValueNode(const NodePtr &node) {
  if (!node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The output of node should be value node, but got " << node->get()->ToString();
  }
  auto value_node = node->get()->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto v = value_node->value();
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<ValueTuple>() && !v->isa<ValueList>()) {
    MS_LOG(EXCEPTION) << "The output of node should be tuple, but got " << node->get()->ToString();
  }
  auto sh = GetValue<std::vector<int64_t>>(v);
  return sh;
}

NodePtr SumGrad(const BpropIRBuilder *ib, const NodePtr &x, const std::vector<int64_t> &axis, const NodePtr &dout) {
  // Grad definition for `Sum` operation.
  auto input_shape = ib->GetShape(x);
  auto output_shape_kept_dims = ReduceShape(input_shape, axis);
  auto tile_scaling = TupleDiv(input_shape, output_shape_kept_dims);
  auto grad = ib->Reshape(dout, output_shape_kept_dims);
  return ib->Emit("Tile", {grad, ib->Value(tile_scaling)});
}

NodePtr MinOrMaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const std::vector<int64_t> &axis, const NodePtr &out,
                     const NodePtr &dout) {
  auto input_shape = ib->GetShape(x);
  auto output_shape_kept_dims = ReduceShape(input_shape, axis);
  auto y = ib->Reshape(out, output_shape_kept_dims);
  auto grad = ib->Reshape(dout, output_shape_kept_dims);
  auto indicators = ib->Cast(ib->Emit("Equal", {y, x}), ib->GetDtype(grad));
  auto minn = 1e-24;
  auto min_num = ib->Tensor(minn, ib->GetDtype(grad));
  auto num_selected =
    ib->Reshape(ib->Emit("ReduceSum", {indicators, ib->Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(false)}}),
                output_shape_kept_dims) +
    min_num;
  return indicators / num_selected * grad;
}

NodePtr ArgminOrArgmaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const int64_t &axis, const bool &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max) {
  auto x_shape = ib->GetShape(x);
  auto x_axis = axis;
  auto onehot_axis = x_axis;
  NodePtr dout_expand;
  NodePtr new_out = out;
  if (keep_dims) {
    dout_expand = ib->TupleGetItem(dout, 1);
    if (is_max) {
      new_out = ib->Emit("ArgMaxWithValue", {x}, {{"axis", MakeValue(axis)}, {"keep_dims", MakeValue(false)}});
    } else {
      new_out = ib->Emit("ArgMinWithValue", {x}, {{"axis", MakeValue(axis)}, {"keep_dims", MakeValue(false)}});
    }
  } else {
    dout_expand = ib->Emit("ExpandDims", {ib->TupleGetItem(dout, 1), ib->Value<int64_t>(onehot_axis)});
  }
  auto out_shape = ib->GetShape(ib->TupleGetItem(new_out, 0));
  if (onehot_axis >= SizeToLong(out_shape.size())) {
    onehot_axis = -1;
  }
  auto type_x = ib->GetDtype(x);
  auto on_value = ib->Tensor(1, type_x);
  auto off_value = ib->Tensor(0, type_x);
  int64_t depth = x_shape[axis];
  auto dx =
    dout_expand * ib->Emit("OneHot", {ib->TupleGetItem(new_out, 0), ib->Value<int64_t>(depth), on_value, off_value},
                           {{"axis", MakeValue(onehot_axis)}});
  return dx;
}

TypeId PromoteBinaryDtype(TypeId t1, TypeId t2) {
  if (t1 == t2) {
    return t1;
  }
  static std::unordered_set<TypeId> complex_types{kNumberTypeComplex64, kNumberTypeComplex128};
  return GetOutputDtype(
    t1, t2, (complex_types.find(t1) != complex_types.end() || complex_types.find(t2) != complex_types.end()));
}
}  // namespace mindspore::expander::bprop
