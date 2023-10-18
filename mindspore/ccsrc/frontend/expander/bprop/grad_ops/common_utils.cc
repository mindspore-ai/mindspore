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
#include "frontend/expander/bprop/grad_ops/common_utils.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
NodePtrList ReturnZeros(BpropIRBuilder *ib) {
  const auto &inputs = ib->GetInputs();
  if (inputs.size() <= kDim2) {
    MS_LOG(EXCEPTION) << "Bprop's inputs size should be greater than 2 (includes out and dout), but got "
                      << inputs.size();
  }
  auto output_num = inputs.size() - kDim2;
  NodePtrList outputs(output_num);
  for (size_t i = 0; i < output_num; ++i) {
    outputs[i] = ib->OutZeros(inputs[i]);
  }
  return outputs;
}

namespace {
std::pair<std::vector<bool>, std::vector<std::vector<int64_t>>> DynBroadcastGradientArgs(
  const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  ShapeVector shape[kDim2] = {x_shape, y_shape};
  auto n = std::max(x_size, y_size);
  std::vector<bool> need_shapecalc = {false, false};
  std::vector<std::vector<int64_t>> reduce_axis(kDim2);
  if (IsDynamicRank(shape[0]) || IsDynamicRank(shape[1])) {
    return {{true, true}, reduce_axis};
  }
  for (size_t i = n; i >= 1; i--) {
    int64_t dim_value[2] = {x_size < i ? 1 : shape[0][x_size - i], y_size < i ? 1 : shape[1][y_size - i]};
    const int64_t reduce_idx = SizeToLong(n - i);
    if (dim_value[1] == dim_value[0]) {
      if (dim_value[0] == -1) {
        need_shapecalc[0] = need_shapecalc[1] = true;
        break;
      }
    } else if (dim_value[1] > 0 && dim_value[0] > 0) {
      for (size_t j = 0; j < kDim2; j++) {
        if (dim_value[j] == 1) {
          (void)reduce_axis[j].emplace_back(reduce_idx);
        }
      }
    } else {
      for (size_t j = 0; j < kDim2; j++) {
        if (dim_value[j] == -1) {
          if (dim_value[j ^ 1] == 1) {
            (void)reduce_axis[j ^ 1].emplace_back(reduce_idx);
          } else {
            need_shapecalc[j] = true;
            if (need_shapecalc[j ^ 1] == need_shapecalc[j]) {
              break;
            }
            (void)reduce_axis[j].emplace_back(reduce_idx);
          }
        }
      }
    }
  }
  return {need_shapecalc, reduce_axis};
}

NodePtrList DynBinopGradCommon(BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                               const NodePtr &dy, size_t shift = 0UL) {
  NodePtr inputs[] = {x, y};
  NodePtrList reduce = {dx, dy};
  ShapeVector shape[] = {ib->GetShape(inputs[0]), ib->GetShape(inputs[1])};
  auto [need_shapecalc, reduce_axis] = DynBroadcastGradientArgs(shape[0], shape[1]);
  NodePtrList broadcast_axes;
  if (need_shapecalc[0] || need_shapecalc[1]) {
    broadcast_axes = ib->BroadcastGradientArgs(inputs[0], inputs[1], shift);
  }
  for (size_t i = 0; i < kDim2; i++) {
    auto dout_shape = ib->GetShape(reduce[i]);
    if (!need_shapecalc[i] && IsDynamicRank(dout_shape)) {
      MS_LOG(WARNING) << "The dynamic shape inference of" << reduce[i]->get()->ToString() << " is overly generalized.";
    }
    if (!need_shapecalc[i] && !IsDynamicRank(dout_shape)) {
      if (!reduce_axis[i].empty()) {
        reduce[i] =
          ib->ReduceSum(reduce[i], ib->Value<ShapeVector>(reduce_axis[i]), dout_shape.size() == shape[i].size(), true);
      }
      if (ib->GetRank(reduce[i]) != shape[i].size()) {
        reduce[i] = ib->Reshape(reduce[i], ib->Shape(inputs[i]));
      }
    } else {
      bool keep_dims = (!IsDynamicRank(shape[0]) && !IsDynamicRank(shape[1]) && shape[i].size() >= shape[i ^ 1].size());
      reduce[i] = ib->ReduceSum(reduce[i], broadcast_axes[i], keep_dims, true);
      reduce[i] = ib->Reshape(reduce[i], ib->Shape(inputs[i]));
    }
  }
  return reduce;
}

TypeId GetOutputDtype(TypeId t1, TypeId t2, bool use_complex = false) {
  static std::unordered_map<TypeId, int> complex_priority_map{
    {kNumberTypeFloat32, 0}, {kNumberTypeFloat32, 1}, {kNumberTypeComplex64, 2}, {kNumberTypeComplex128, 4}};
  static std::unordered_map<TypeId, int> type_priority_map{
    {kNumberTypeBool, 0},     {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},     {kNumberTypeUInt16, 3},
    {kNumberTypeInt16, 4},    {kNumberTypeUInt32, 5},  {kNumberTypeInt32, 6},    {kNumberTypeUInt64, 7},
    {kNumberTypeInt64, 8},    {kNumberTypeFloat16, 9}, {kNumberTypeFloat32, 10}, {kNumberTypeFloat64, 11},
    {kNumberTypeBFloat16, 12}};
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

int64_t NormalizeAxis(int64_t axis, size_t rank) {
  auto rank_i = SizeToLong(rank);
  if (axis < -rank_i || axis >= rank_i) {
    MS_EXCEPTION(ValueError) << "For rank " << rank << ", the axis must be in range [" << -rank_i << ", " << rank_i
                             << "), but got " << axis;
  }
  return (axis < 0) ? (axis + rank_i) : axis;
}

std::pair<ShapeVector, ShapeVector> SplitShapeIndex(const ShapeVector &input_shape, const ShapeVector &axis) {
  auto rank = SizeToLong(input_shape.size());
  if (rank == 0) {
    return {};
  }
  std::vector<bool> reduction_indices_map(input_shape.size());
  ShapeVector perm;
  int64_t reduced_num = 1;
  int64_t other_num = 1;
  for (auto i : axis) {
    if (i < 0) {
      i += rank;
    }
    reduction_indices_map[i] = True;
    reduced_num *= input_shape[LongToSize(i)];
    (void)perm.emplace_back(i);
  }
  for (int64_t i = 0; i < rank; i++) {
    if (!reduction_indices_map[i]) {
      other_num *= input_shape[LongToSize(i)];
      (void)perm.emplace_back(i);
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
  std::vector<int64_t> grad_x_reduce_idx;
  std::vector<int64_t> grad_y_reduce_idy;
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  auto n = std::max(x_size, y_size);
  for (size_t i = n; i >= 1; i--) {
    auto x_i = x_size < i ? 1 : x_shape[x_size - i];
    auto y_i = y_size < i ? 1 : y_shape[y_size - i];
    const int64_t reduce_idx = SizeToLong(n - i);
    if (x_i == y_i) {
      continue;
    } else if (x_i == 1) {
      grad_x_reduce_idx.push_back(reduce_idx);
    } else if (y_i == 1) {
      grad_y_reduce_idy.push_back(reduce_idx);
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input(" << x_shape << ", " << y_shape
                        << ") for BroadcastGradientArgs.";
    }
  }

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
  if (x.empty()) {
    return {};
  }
  if (axis.empty()) {
    return std::vector<int64_t>(x.size(), 1LL);
  }
  int64_t x_rank = SizeToLong(x.size());
  std::vector<int64_t> out(x);
  for (auto i : axis) {
    if (i >= x_rank || i < (-x_rank)) {
      MS_LOG(EXCEPTION) << "axis should be in range [" << (-x_rank) << ", " << x_rank << ").";
    }
    if (i < 0) {
      i += x_rank;
    }
    out[i] = 1LL;
  }
  return out;
}

int64_t GetIntValue(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto real_node = node->get();
  MS_EXCEPTION_IF_NULL(real_node);
  if (real_node->isa<ValueNode>()) {
    MS_EXCEPTION_IF_NULL(real_node->abstract());
    if (real_node->abstract()->isa<abstract::AbstractTensor>()) {
      auto value_node = real_node->cast<ValueNodePtr>();
      auto t_vec = CheckAndConvertUtils::CheckTensorIntValue("tensor", value_node->value(), "bprop");
      MS_EXCEPTION_IF_CHECK_FAIL(t_vec.size() >= kIndex1, "Get single tensor value failed");
      return t_vec[kIndex0];
    }
  }
  return AnfUtils::GetIntValue(node->BuildValue());
}

std::vector<int64_t> GetIntList(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->data_sync();
    return CheckAndConvertUtils::CheckTensorIntValue("tensor", value, "bprop");
  } else {
    return CheckAndConvertUtils::CheckIntOrTupleInt("value", value, "bprop");
  }
}

std::vector<int64_t> GetIntList(const NodePtr &node) {
  auto value = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  return GetIntList(value);
}

NodePtrList BinopGradCommon(BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                            const NodePtr &dy, size_t shift) {
  // Common grad definition for binary operations with shift.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  NodePtr inputs[] = {x, y};
  ShapeVector shape[] = {ib->GetShape(inputs[0]), ib->GetShape(inputs[1])};
  NodePtrList reduce = {dx, dy};
  if (IsDynamicRank(shape[0]) || IsDynamicRank(shape[1])) {
    return DynBinopGradCommon(ib, x, y, dx, dy, shift);
  }
  if (shape[kIndex0].size() <= shift && shape[kIndex0].size() == shape[kIndex1].size()) {
    return reduce;
  }
  ShapeVector broadcast_shape[kDim2];
  for (size_t i = 0; i < kDim2; i++) {
    broadcast_shape[i] = ShapeVector(shape[i].begin(), shape[i].end() - shift);
  }

  if (broadcast_shape[0].empty() || broadcast_shape[1].empty()) {
    for (size_t i = 0; i < kDim2; i++) {
      if (broadcast_shape[i].empty()) {
        if (shift) {
          std::vector<int64_t> axis(broadcast_shape[i ^ 1].size());
          std::iota(axis.begin(), axis.begin(), 0LL);
          reduce[i] = ib->ReduceSum(reduce[i], axis);
        } else {
          reduce[i] = ib->ReduceSum(reduce[i]);
        }
      }
    }
  } else if (!IsDynamic(broadcast_shape[0]) && !IsDynamic(broadcast_shape[1])) {
    std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(broadcast_shape[0], broadcast_shape[1]);
    for (size_t i = 0; i < kDim2; i++) {
      if (!bc_axis[i].empty()) {
        reduce[i] = ib->ReduceSum(reduce[i], bc_axis[i], ib->GetRank(reduce[i]) == shape[i].size());
      }
      if (ib->GetRank(reduce[i]) != shape[i].size()) {
        reduce[i] = ib->Reshape(reduce[i], shape[i]);
      }
    }
  } else {
    return DynBinopGradCommon(ib, x, y, dx, dy, shift);
  }
  return reduce;
}

std::vector<int64_t> Range(int64_t start, int64_t stop, int64_t step) {
  if (step == 0) {
    MS_EXCEPTION(ValueError) << "For Range, step should not be 0";
  }
  auto size = stop - start;
  if (size * step <= 0) {
    return {};
  }
  if (size % step == 0) {
    size = size / step;
  } else {
    size = size / step + 1;
  }
  std::vector<int64_t> range(LongToSize(size));
  for (size_t i = 0; i < range.size(); i++, start += step) {
    range[i] = start;
  }
  return range;
}

std::vector<int64_t> Range(int64_t stop) { return Range(0, stop); }

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  std::vector<int64_t> reverse_axis;
  if (x_shape.empty()) {
    return reverse_axis;
  }
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  reverse_axis.reserve(x_shape.size());
  for (int64_t i = 0; i < rk; ++i) {
    (void)reverse_axis.emplace_back(i);
  }
  reverse_axis[LongToSize(axis)] = rk - 1;
  reverse_axis[LongToSize(rk - 1)] = axis;
  return reverse_axis;
}

int64_t CheckRange(int64_t idx, int64_t dim_size) {
  if (idx < -dim_size || idx >= dim_size) {
    MS_EXCEPTION(IndexError) << "index {" << idx << "} is out of bounds for dimension with size {" << dim_size << "}";
  }
  return idx < 0 ? (idx + dim_size) : idx;
}

NodePtr GetEps(BpropIRBuilder *ib, const TypePtr &type) {
  constexpr auto epsilon = 0.000977;
  switch (type->type_id()) {
    case kNumberTypeFloat16:
      return ib->Tensor(epsilon, type);
    case kNumberTypeFloat32:
      return ib->Tensor(std::numeric_limits<float>::epsilon(), type);
    case kNumberTypeFloat64:
      return ib->Tensor(std::numeric_limits<double>::epsilon(), type);
    default:
      return ib->Tensor(0, type);
  }
}

std::vector<int64_t> GenerateInverseIndex(const std::vector<int64_t> &x_shp, int64_t axis_v, int64_t batch_dims) {
  int64_t x_rank = static_cast<int64_t>(x_shp.size());
  auto index = Range(x_rank);
  if (axis_v < 0) {
    axis_v += x_rank;
  }
  std::vector<int64_t> perm;
  auto start1 = x_rank <= 1 ? index.end() : index.begin() + batch_dims + 1;
  auto end1 = axis_v + 1 >= x_rank ? index.end() : index.begin() + axis_v + 1;
  auto start2 = axis_v + 1 >= x_rank ? index.end() : index.begin() + axis_v + 1;
  (void)std::copy(index.begin(), index.begin() + batch_dims, std::back_inserter(perm));
  (void)std::copy(start1, end1, std::back_inserter(perm));
  perm.push_back(batch_dims);
  (void)std::copy(start2, index.end(), std::back_inserter(perm));
  return perm;
}

std::vector<int64_t> GenerateShapeIndex(const std::vector<int64_t> &out_shp, const std::vector<int64_t> &ind_shp,
                                        int64_t axis_v, int64_t batch_dims) {
  int64_t out_rank = static_cast<int64_t>(out_shp.size());
  int64_t ind_rank = static_cast<int64_t>(ind_shp.size());
  if (axis_v < 0) {
    axis_v += out_rank - ind_rank + 1;
  }
  auto perm_part1 = Range(axis_v, axis_v + ind_rank - batch_dims);
  auto index = Range(out_rank);
  std::vector<int64_t> perm;
  auto end = axis_v >= out_rank ? out_rank - 1 : axis_v;
  auto start =
    (axis_v + ind_rank - batch_dims) >= out_rank ? index.end() : (index.begin() + axis_v + ind_rank - batch_dims);
  (void)std::copy(index.begin(), index.begin() + batch_dims, std::back_inserter(perm));
  (void)std::copy(perm_part1.begin(), perm_part1.end(), std::back_inserter(perm));
  (void)std::copy(index.begin() + batch_dims, index.begin() + end, std::back_inserter(perm));
  (void)std::copy(start, index.end(), std::back_inserter(perm));
  return perm;
}

std::vector<int64_t> RegenerateOutputShape(const std::vector<int64_t> &x_shp, const std::vector<int64_t> &ind_shp,
                                           int64_t axis_v, int64_t batch_dims) {
  int64_t rank = static_cast<int64_t>(x_shp.size());
  if (axis_v < 0) {
    axis_v += rank;
  }
  std::vector<int64_t> out_shp;
  auto end = axis_v >= rank ? rank - 1 : axis_v;
  auto start = axis_v + 1 >= rank ? x_shp.end() : x_shp.begin() + axis_v + 1;
  (void)std::copy(x_shp.begin(), x_shp.begin() + end, std::back_inserter(out_shp));
  (void)std::copy(ind_shp.begin() + batch_dims, ind_shp.end(), std::back_inserter(out_shp));
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
  auto res_sz = static_cast<size_t>(2 * max_len);
  res.reserve(res_sz);
  while (i < max_len && j < max_len) {
    auto idx_i = LongToSize(i);
    auto idx_j = LongToSize(j);
    if (len_cmp == 0) {
      res.push_back(multiples[idx_i]);
      res.push_back(shapex[idx_j]);
      i++;
      j++;
    } else if (len_cmp > 0) {
      res.push_back(multiples[idx_i]);
      res.push_back(1);
      i++;
      len_cmp--;
    } else {
      res.push_back(1);
      res.push_back(shapex[idx_j]);
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
    auto idx = LongToSize(i);
    if (check_perm[idx] != i) {
      MS_LOG(EXCEPTION) << "For InvertPermutation, the input_x should be '[0-" << (perm_size - 1) << "]', but got "
                        << check_perm;
    }
    res[LongToSize(perm[idx])] = i;
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
  (void)trans.insert(trans.end(), after_axis.begin(), after_axis.end());
  trans.push_back(axis);
  return trans;
}

DEF_PURE_SHAPE_CALC(reduce_shape_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    auto axis_value = inputs.at(1);
    auto r_shape = ReduceShape(x_shape, axis_value);
    return {r_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    return {IsDynamicRank(inputs.at(0)) ? -1 : static_cast<int64_t>(inputs.at(0).size())};
  });
NodePtr SumGrad(BpropIRBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &dout, const bool keep_dims) {
  auto grad = dout;
  if (!keep_dims) {
    auto calc_res = ib->ShapeCalc(reduce_shape_shapecalc, {x, axis}, {1});
    grad = ib->Reshape(grad, calc_res[0]);
  }
  return ib->BroadcastTo(grad, x);
}

NodePtr MinOrMaxGrad(BpropIRBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &out,
                     const NodePtr &dout) {
  auto y = out;
  auto grad = dout;
  if (!ib->GetAttr<bool>("keep_dims")) {
    auto output_shape_kept_dims = ib->ShapeCalc(reduce_shape_shapecalc, {x, axis}, {1})[0];
    y = ib->Reshape(out, output_shape_kept_dims);
    grad = ib->Reshape(dout, output_shape_kept_dims);
  }
  auto indicators = ib->Cast(ib->Equal(y, x), ib->GetDtype(grad));
  auto num_selected = ib->ReduceSum(indicators, axis, true, false);
  return indicators / num_selected * grad;
}

class ArgminOrArgmaxShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_ArgminOrArgmax", ArgminOrArgmaxShapeCalc)
  explicit ArgminOrArgmaxShapeCalc(int64_t axis) : ShapeCalcFunctor("ShapeCalc_ArgminOrArgmax"), axis_(axis) {}
  ValuePtr ToValue() const override { return MakeValue(axis_); }
  void FromValue(const ValuePtr &value) override { axis_ = GetValue<int64_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto indices_expand_rank = inputs.at(0).size();
    auto x_shape = inputs.at(1);
    std::vector<int64_t> broad_shape(indices_expand_rank, 1);
    auto x = LongToSize(axis_ + SizeToLong(indices_expand_rank));
    auto depth = x_shape[x];
    broad_shape[x] = depth;
    return {broad_shape, {depth}};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t x_rank = IsDynamicRank(inputs.at(0)) ? -1 : static_cast<int64_t>(inputs.at(0).size());
    return {x_rank, 1};
  }

 protected:
  int64_t axis_{0};
};
REG_FUNCTOR("ShapeCalc_ArgminOrArgmax", ArgminOrArgmaxShapeCalc);

NodePtr ArgminOrArgmaxGrad(BpropIRBuilder *ib, const NodePtr &x, const int64_t &axis, const bool &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max) {
  auto x_shape = ib->GetShape(x);
  int64_t x_axis = axis;
  if (!IsDynamicRank(x_shape)) {
    x_axis = CheckRange(axis, SizeToLong(x_shape.size()));
  } else if (axis < 0) {
    MS_LOG_EXCEPTION << "For ArgminOrArgmaxGrad, when axis is negative,"
                     << "input_x is currently not supported as a dynamic rank.";
  }
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
    dout_expand = ib->Emit("ExpandDims", {ib->TupleGetItem(dout, 1), ib->Value<int64_t>(x_axis)});
  }
  auto type_x = ib->GetDtype(x);
  auto on_value = ib->Tensor(1, type_x);
  auto off_value = ib->Tensor(0, type_x);
  auto out_0 = ib->TupleGetItem(new_out, 0);
  NodePtr depth = ib->Value<int64_t>(1);
  if (!x_shape.empty()) {
    depth = ib->TupleGetItem(ib->Shape(x), LongToSize(x_axis));
  }
  if (x_axis >= 0) {
    auto onehot_axis = x_axis;
    auto out_shape = ib->GetShape(out_0);
    if (!IsDynamic(out_shape) && onehot_axis >= SizeToLong(out_shape.size())) {
      onehot_axis = -1;
    }
    auto dx = dout_expand * ib->Emit("OneHot", {out_0, depth, on_value, off_value}, {{"axis", MakeValue(onehot_axis)}});
    if (x_shape.empty()) {
      dx = ib->Emit("Squeeze", {dx});
    }
    return dx;
  } else {
    auto indices_expand = ib->ExpandDims(out_0, x_axis);
    auto res = ib->ShapeCalc(std::make_shared<ArgminOrArgmaxShapeCalc>(x_axis), {indices_expand, x});
    auto broad_shape = res[0];
    depth = res[1];
    auto depth_range = ib->Range(depth);
    auto depth_broad = ib->Reshape(depth_range, broad_shape);
    auto one_hot_bool = ib->Equal(indices_expand, depth_broad);
    auto one_hot_res = ib->Cast(one_hot_bool, type_x);
    return dout_expand * one_hot_res;
  }
}

TypeId PromoteBinaryDtype(TypeId t1, TypeId t2) {
  if (t1 == t2) {
    return t1;
  }
  static std::unordered_set<TypeId> complex_types{kNumberTypeComplex64, kNumberTypeComplex128};
  return GetOutputDtype(
    t1, t2, (complex_types.find(t1) != complex_types.end() || complex_types.find(t2) != complex_types.end()));
}

NodePtr LGamma(BpropIRBuilder *ib, const NodePtr &x) {
  auto k_lanczos_gamma = 7;
  auto k_base_lanczos_coeff = 0.9999999999998099;
  double k_lanczos_coefficients[8] = {676.520368121885098567009190444019, -1259.13921672240287047156078755283,
                                      771.3234287776530788486528258894,   -176.61502916214059906584551354,
                                      12.507343278686904814458936853,     -0.13857109526572011689554707,
                                      9.984369578019570859563e-6,         1.50563273514931155834e-7};
  auto input_dtype = ib->GetDtype(x);
  auto one_half = ib->Tensor(0.5, input_dtype);
  auto one = ib->Tensor(1, input_dtype);
  auto zero = ib->Tensor(0, input_dtype);
  auto log_sqrt_two_pi = ib->Tensor((log_2 + log_pi) / 2, input_dtype);
  auto lanczos_gamma_plus_one_half = k_lanczos_gamma + 0.5;
  auto log_lanczos_gamma_plus_one_half = log(lanczos_gamma_plus_one_half);
  auto inf = std::numeric_limits<double>::infinity();
  auto infinity = ib->Fill(inf, ib->Shape(x), input_dtype->type_id());
  auto need_to_reflect = ib->Less(x, one_half);
  auto neg_input = ib->Neg(x);
  auto z = ib->Select(need_to_reflect, neg_input, ib->Sub(x, one));
  auto CalculateReflectedX = [&ib, &z, &k_base_lanczos_coeff, &k_lanczos_coefficients]() -> NodePtr {
    auto z_dtype = ib->GetDtype(z);
    NodePtr reflex_x = ib->Tensor(k_base_lanczos_coeff, z_dtype);
    for (int i = 0; i < 8; ++i) {
      auto btmp = ib->Add(z, ib->Tensor(i, z_dtype));
      btmp = ib->Add(btmp, (ib->Tensor(1, z_dtype)));
      auto product = ib->RealDiv((ib->Tensor(k_lanczos_coefficients[i], z_dtype)), btmp);
      reflex_x = ib->Add(product, reflex_x);
    }
    return reflex_x;
  };
  auto reflex_x = CalculateReflectedX();
  auto lanczos_tensor = ib->Tensor(lanczos_gamma_plus_one_half, input_dtype);
  auto log_lanczos_tensor = ib->Tensor(log_lanczos_gamma_plus_one_half, input_dtype);
  auto t = ib->Add(z, lanczos_tensor);
  auto log_t = ib->Add((ib->Emit("Log1p", {ib->RealDiv(z, lanczos_tensor)})), log_lanczos_tensor);
  auto log_y = ib->Add(
    (ib->Add((ib->Log(reflex_x)), (ib->Mul((ib->Sub((ib->Add(z, one_half)), (ib->RealDiv(t, log_t)))), log_t)))),
    log_sqrt_two_pi);
  auto abs_input = ib->Emit("Abs", {x});
  auto abs_frac_input = ib->Sub(abs_input, (ib->Emit("Floor", {abs_input})));
  auto new_x = ib->Select(ib->LessEqual(x, zero), ib->Select(ib->Equal(abs_frac_input, zero), infinity, x), x);
  auto reduced_frac_input =
    ib->Select(ib->Greater(abs_frac_input, one_half), ib->Sub(one, abs_frac_input), abs_frac_input);
  auto reflection_denom =
    ib->Log(ib->Emit("Sin", {ib->Mul(ib->Tensor(pi, ib->GetDtype(reduced_frac_input)), reduced_frac_input)}));
  auto reflection =
    ib->Select(ib->Emit("IsFinite", {reflection_denom}),
               ib->Add((ib->Sub((ib->Neg(reflection_denom)), log_y)), ib->Tensor(log_pi, ib->GetDtype(log_y))),
               ib->Neg(reflection_denom));
  auto result = ib->Select(need_to_reflect, reflection, log_y);
  return ib->Select(ib->Emit("IsFinite", {new_x}), result, infinity);
}

bool CheckType(const TypePtr &check_type, const std::set<TypePtr> &template_types) {
  return std::any_of(template_types.begin(), template_types.end(), [&check_type](const TypePtr &accept) -> bool {
    return IsIdentidityOrSubclass(check_type, accept);
  });
}

ShapeVector PoolToNHWC(const ShapeVector &v) {
  ShapeVector new_v(v);
  new_v[kIndex1] = v[kIndex2];
  new_v[kIndex2] = v[kIndex3];
  new_v[kIndex3] = v[kIndex1];
  return new_v;
}
ShapeVector ConvToNHWC(const ShapeVector &v) {
  ShapeVector new_v(v);
  new_v[kIndex0] = v[kIndex1];
  new_v[kIndex1] = v[kIndex2];
  new_v[kIndex2] = v[kIndex3];
  new_v[kIndex3] = 1;
  return new_v;
}

ShapeVector GetShapeByRange(const ShapeVector &v, int64_t begin, int64_t end) {
  // Get range [begin, end) in v.
  auto rank = SizeToLong(v.size());
  auto real_begin = std::min((begin < 0) ? (rank + begin) : begin, rank);
  auto real_end = std::min((end < 0) ? (rank + end) : end, rank);
  ShapeVector res(v.begin() + real_begin, v.begin() + real_end);
  return res;
}

NodePtr MatrixTranspose(BpropIRBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  if (IsDynamicRank(shape)) {
    auto dim = ib->Emit("Rank", {x});
    auto perm = ib->Range(ib->Tensor(0, kInt64), ib->Emit("Cast", {dim, ib->EmitValue(kInt64)}), ib->Tensor(1, kInt64));
    auto stridedslice_helper = [&perm, &ib](const NodePtr &x) {
      return ib->Emit("StridedSlice",
                      {perm, ib->TupleGetItem(x, ib->Value(static_cast<int64_t>(0))),
                       ib->TupleGetItem(x, ib->Value(static_cast<int64_t>(1))),
                       ib->TupleGetItem(x, ib->Value(static_cast<int64_t>(2)))},
                      {{"begin_mask", MakeValue<int64_t>(0LL)},
                       {"end_mask", MakeValue<int64_t>(0LL)},
                       {"ellipsis_mask", MakeValue<int64_t>(0LL)},
                       {"new_axis_mask", MakeValue<int64_t>(0LL)},
                       {"shrink_axis_mask", MakeValue<int64_t>(0LL)}});
    };
    auto normalize_slice_helper = [&perm, &ib](int64_t x, int64_t y, int64_t z,
                                               const std::vector<int64_t> &init_by_none) {
      return ib->Emit("NormalizeSlice",
                      {perm, ib->Value(static_cast<int64_t>(x)), ib->Value(static_cast<int64_t>(y)),
                       ib->Value(static_cast<int64_t>(z))},
                      {{kAttrTupleIndexAxis, MakeValue(static_cast<int64_t>(0))},
                       {kAttrTupleIndexTypes, MakeValue({})},
                       {kAttrExpandDimsMask, MakeValue(static_cast<int64_t>(0))},
                       {kAttrInitByNone, MakeValue(init_by_none)}});
    };
    auto range_1 = normalize_slice_helper(0, -2, 0, {1, 0, 1});
    auto range_2 = normalize_slice_helper(-1, 0, 0, {0, 1, 1});
    auto range_3 = normalize_slice_helper(-2, -1, 0, {0, 0, 1});
    auto part_1 = stridedslice_helper(range_1);
    auto part_2 = stridedslice_helper(range_2);
    auto part_3 = stridedslice_helper(range_3);
    perm = ib->Concat({part_1, part_2, part_3}, -1);
    return ib->Transpose(x, perm);
  }
  auto dim = shape.size();
  if (dim < kDim2) {
    MS_LOG_EXCEPTION << "For MatrixTranspose, input's ndim " << dim << " is less or equal to 2, which is invalid";
  }
  std::vector<int64_t> perm(dim);
  for (size_t i = 0; i < dim; i++) {
    perm[i] = static_cast<int64_t>(i);
  }
  std::swap(perm[dim - kIndex2], perm[dim - kIndex1]);
  return ib->Transpose(x, perm);
}

NodePtr Adjoint(BpropIRBuilder *ib, const NodePtr &x) { return MatrixTranspose(ib, ib->Conj(x)); }
}  // namespace mindspore::expander::bprop
