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
#include "frontend/expander/grad/grad_utils.h"

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
#include "ops_utils/op_utils.h"

namespace mindspore::expander::bprop {
NodePtrList ReturnZeros(BpropBuilder *ib) {
  const auto &inputs = ib->GetInputs();
  if (inputs.size() <= i2) {
    MS_LOG(EXCEPTION) << "Bprop's inputs size should be greater than 2 (includes out and dout), but got "
                      << inputs.size();
  }
  auto output_num = inputs.size() - i2;
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
  ShapeVector shape[i2] = {x_shape, y_shape};
  auto n = std::max(x_size, y_size);
  std::vector<bool> need_shapecalc = {false, false};
  std::vector<std::vector<int64_t>> reduce_axis(i2);
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
      for (size_t j = 0; j < i2; j++) {
        if (dim_value[j] == 1) {
          (void)reduce_axis[j].emplace_back(reduce_idx);
        }
      }
    } else {
      for (size_t j = 0; j < i2; j++) {
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

NodePtrList DynBinopGradCommon(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                               const NodePtr &dy, size_t shift = 0UL) {
  NodePtr inputs[] = {x, y};
  NodePtrList reduce = {dx, dy};
  ShapeVector shape[] = {ib->GetShape(inputs[0]), ib->GetShape(inputs[1])};
  auto [need_shapecalc, reduce_axis] = DynBroadcastGradientArgs(shape[0], shape[1]);
  NodePtrList broadcast_axes;
  if (need_shapecalc[0] || need_shapecalc[1]) {
    broadcast_axes = ib->BroadcastGradientArgs(inputs[0], inputs[1], shift);
  }
  for (size_t i = 0; i < i2; i++) {
    if (reduce[i] == nullptr) {
      continue;
    }
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

std::vector<int64_t> ReduceShapeTupleDiv(const std::vector<int64_t> &x, const std::vector<int64_t> &y) {
  std::vector<int64_t> out;
  if (x.size() != y.size()) {
    MS_LOG(EXCEPTION) << "The size of inputs of ReduceShapeTupleDiv must be the same, but the size of divisor tuple is"
                      << " " << y.size() << ", the size of dividend tuple is " << x.size() << ".";
  }
  for (size_t i = 0; i < y.size(); i++) {
    if (x[i] == 0 && y[i] == 0) {
      out.push_back(1LL);
      continue;
    }
    if (y[i] == 0) {
      MS_LOG(EXCEPTION) << "The divisor value should not be 0!";
    }
    if ((x[i] % y[i]) != 0) {
      MS_LOG(EXCEPTION) << "The inputs of ReduceShapeTupleDiv should be divisible, but they are not divisible now, "
                        << "the dividend is " << x[i] << ", the divisor is " << y[i] << ".";
    }
    out.push_back(x[i] / y[i]);
  }
  return out;
}

std::vector<int64_t> ReduceShape(const std::vector<int64_t> &x, const std::vector<int64_t> &axis, bool skip_mode) {
  if (x.empty()) {
    return {};
  }
  if (axis.empty()) {
    if (skip_mode) {
      return x;
    }
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
  auto value = node->BuildValue();
  if (value->isa<tensor::Tensor>()) {
    auto t_vec = CheckAndConvertUtils::CheckTensorIntValue("tensor", value, "bprop");
    MS_EXCEPTION_IF_CHECK_FAIL(t_vec.size() >= i1, "Get single tensor value failed");
    return t_vec[i0];
  }
  return AnfUtils::GetIntValue(value);
}

NodePtr StaticBinopGradCommon(BpropBuilder *ib, const NodePtr &dx, const ShapeArray &shape,
                              const ShapeArray &broadcast_shape, size_t shift, size_t index, bool *is_dynamic_shape) {
  NodePtr reduce_dx = dx;
  auto shape_dynamic_dims = std::count_if(shape[index].begin(), shape[index].end(), [](int64_t x) { return x <= -1; });
  if (broadcast_shape[i0].empty() || broadcast_shape[i1].empty()) {
    if (broadcast_shape[index].empty()) {
      if (shift) {
        std::vector<int64_t> axis(broadcast_shape[index ^ 1].size());
        std::iota(axis.begin(), axis.end(), 0LL);
        reduce_dx = ib->SumExt(reduce_dx, ib->Value<ShapeVector>(axis), ib->Value(false));
      } else {
        reduce_dx = ib->SumExt(reduce_dx, ib->EmitValue(kNone), ib->Value(false));
      }
    }
  } else if (!IsDynamic(broadcast_shape[0]) && !IsDynamic(broadcast_shape[1]) && shape_dynamic_dims <= 1) {
    std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgsInferValue(broadcast_shape[0], broadcast_shape[1]);
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

NodePtrList BinopGradCommon(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx, const NodePtr &dy,
                            size_t shift) {
  // Common grad definition for binary operations with shift.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  NodePtrList inputs{x, y};
  ShapeArray shape{ib->GetShape(inputs[i0]), ib->GetShape(inputs[i1])};
  NodePtrList reduce = {dx, dy};
  if (IsDynamicRank(shape[i0]) || IsDynamicRank(shape[i1])) {
    return DynBinopGradCommon(ib, x, y, dx, dy, shift);
  }
  if (shape[i0].size() <= shift && shape[i0].size() == shape[i1].size()) {
    return reduce;
  }
  ShapeArray broadcast_shape(i2);
  for (size_t i = 0; i < i2; i++) {
    broadcast_shape[i] = ShapeVector(shape[i].begin(), shape[i].end() - shift);
  }
  bool is_x_shape_dynamic = false;
  bool is_y_shape_dynamic = false;
  if (dx != nullptr) {
    reduce[i0] = StaticBinopGradCommon(ib, reduce[i0], shape, broadcast_shape, shift, i0, &is_x_shape_dynamic);
  }
  if (dy != nullptr) {
    reduce[i1] = StaticBinopGradCommon(ib, reduce[i1], shape, broadcast_shape, shift, i1, &is_y_shape_dynamic);
  }
  if (is_x_shape_dynamic || is_y_shape_dynamic) {
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

NodePtr GetEps(BpropBuilder *ib, const TypePtr &type) {
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

class ReduceShapeShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_ReduceShape", ReduceShapeShapeCalc)
  explicit ReduceShapeShapeCalc(bool skip_mode) : ShapeCalcFunctor("ShapeCalc_ReduceShape"), skip_mode_(skip_mode) {}
  ValuePtr ToValue() const override { return MakeValue(skip_mode_); }
  void FromValue(const ValuePtr &value) override { skip_mode_ = GetValue<bool>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs.at(0);
    auto axis_value = inputs.at(1);
    auto r_shape = ReduceShape(x_shape, axis_value, skip_mode_);
    auto scaling = ReduceShapeTupleDiv(x_shape, r_shape);
    return {r_shape, scaling};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t x_rank = IsDynamicRank(inputs.at(0)) ? -1 : static_cast<int64_t>(inputs.at(0).size());
    return {x_rank, x_rank};
  }

 protected:
  bool skip_mode_ = false;
};
REG_FUNCTOR("ShapeCalc_ReduceShape", ReduceShapeShapeCalc);

NodePtr SumGrad(Emitter *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &dout, bool keep_dims,
                bool skip_mode) {
  auto grad = dout;
  auto calc_res = ib->ShapeCalc(std::make_shared<ReduceShapeShapeCalc>(skip_mode), {x, axis}, {1});
  if (!keep_dims) {
    grad = ib->Reshape(grad, calc_res[0]);
  }
  auto tile_scaling = calc_res[1];
  if (tile_scaling->input_type() == InputType::kConstant || IsDynamic(x->shape())) {
    return ib->Tile(grad, tile_scaling);
  }
  return ib->BroadcastTo(grad, x);
}

// using input.shape to get the unsqueezed outputs
NodePtrList GetUnsqueezeTensor(Emitter *ib, const NodePtr &input, const NodePtr &axis, bool keep_dims,
                               const NodePtrList &outputs) {
  NodePtrList y;
  if (!keep_dims) {
    auto output_shape_kept_dims = ib->ShapeCalc(std::make_shared<ReduceShapeShapeCalc>(), {input, axis}, {1})[0];
    for (auto node : outputs) {
      y.push_back(ib->Reshape(node, output_shape_kept_dims));
    }
  } else {
    y = outputs;
  }
  return y;
}

NodePtr LogSumExpGrad(Emitter *ib, const NodePtr &input, const NodePtr &dim, bool keepdim, const NodePtr &output,
                      const NodePtr &dout) {
  auto unsqueeze_result = GetUnsqueezeTensor(ib, input, dim, keepdim, {output, dout});
  return ib->Mul(ib->Exp(ib->Sub(input, unsqueeze_result[0])), unsqueeze_result[1]);
}

NodePtr InplacePutGrad(Emitter *ib, const NodePtr &index, const NodePtr &source, bool accumulate, const NodePtr &dout,
                       const NodePtr &type) {
  if (accumulate) {
    return dout;
  }
  auto clone_grad = ib->Clone(dout);
  auto grad = ib->InplacePut(clone_grad, index, ib->Zeros(ib->Shape(source), type), ib->Value<bool>(false));
  return grad;
}

inline NodePtr DynamicRankVarImpl(Emitter *ib, const NodePtr &x, const NodePtr &dout, const NodePtr &grad,
                                  const NodePtr &correction, const NodePtr &mean) {
  const float dof_scale = 2.0;
  auto dout_size = ib->Emit("Size", {dout});
  auto x_size = ib->Emit("Size", {x});
  auto used_size = ib->RealDiv(ib->ScalarToTensor(x_size, kFloat32), ib->ScalarToTensor(dout_size, kFloat32));
  auto dof = ib->RealDiv(ib->Tensor(dof_scale, kFloat32), ib->Sub(used_size, ib->ScalarToTensor(correction, kFloat32)));
  dof = ib->Cast(dof, dout->dtype());
  return ib->Mul(ib->Mul(ib->Cast(grad, dout->dtype()), dof), ib->Sub(x, mean));
}

inline NodePtr DynamicGradUnsqueeze(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &dout,
                                    const NodePtr &keepdim, const NodePtr &rank) {
  auto is_keepdim_false = ib->Equal(keepdim, ib->Value<bool>(false));
  auto cond = ib->LogicalAnd(ib->ScalarToTensor(is_keepdim_false, kBool),
                             ib->Greater(ib->ScalarToTensor(rank, kInt64), ib->Tensor(1, kInt64)));
  auto true_branch = [&x, &axis, &dout](Emitter *e) -> NodePtrList {
    auto output_shape_keepdim = e->ShapeCalc(std::make_shared<ReduceShapeShapeCalc>(false), {x, axis}, {1})[0];
    return {e->Reshape(dout, output_shape_keepdim)};
  };
  auto false_branch = [&dout](Emitter *e) -> NodePtrList { return {dout}; };

  return ib->Conditional(cond, true_branch, false_branch);
}

NodePtr VarGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis_node, const NodePtr &dout,
                const NodePtr &correction, const NodePtr &keepdim) {
  const float dof_scale = 2.0;
  NodePtr axis = axis_node;
  if (ib->GetDtype(axis_node)->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }
  NodePtr grad = dout;
  auto dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout)));
  if (IsDynamicRank(ib->GetShape(x))) {
    auto rank = ib->Emit("Rank", {x});
    grad = DynamicGradUnsqueeze(ib, x, axis, dout, keepdim, rank);
    auto is_zero_rank_cond = ib->Emit("scalar_eq", {rank, ib->Value<int64_t>(0)});
    auto var_zero_rank_impl = [&x, &dout, &grad, &correction, &dtype](Emitter *e) -> NodePtrList {
      auto mean = e->MeanExt(x, e->Value<std::vector<int64_t>>({}), e->Value<bool>(false), dtype);
      return {DynamicRankVarImpl(e, x, dout, grad, correction, mean)};
    };
    auto var_impl = [&x, &axis, &dout, &grad, &correction, &dtype](Emitter *e) -> NodePtrList {
      auto mean = e->MeanExt(x, axis, e->Value<bool>(true), dtype);
      return {DynamicRankVarImpl(e, x, dout, grad, correction, mean)};
    };
    return ib->Conditional(is_zero_rank_cond, var_zero_rank_impl, var_impl);
  } else {
    NodePtr dof = nullptr;
    float dof_imm = 0.0;
    if (IsDynamic(ib->GetShape(x)) || IsDynamic(ib->GetShape(dout))) {
      auto dout_size = ib->DynSize(dout, kFloat32);
      auto used_size = ib->DynSize(x, kFloat32) / dout_size;
      dof = ib->RealDiv(ib->Tensor(dof_scale, kFloat32), ib->Sub(used_size, ib->ScalarToTensor(correction, kFloat32)));
      dof = ib->Cast(dof, ib->GetDtype(dout));
    } else {
      auto dout_size = ib->GetSize(dout);
      if (dout_size == 0) {
        MS_EXCEPTION(ValueError) << "For 'Var', out shape size can not be 0";
      }
      auto used_size = ib->GetSize(x) / dout_size;
      dof_imm = dof_scale / (used_size - GetValue<int64_t>(correction->BuildValue()));
    }
    auto rank = ib->GetShape(x).size();
    auto mean = rank == 0 ? ib->MeanExt(x, ib->Value<std::vector<int64_t>>({}), ib->Value<bool>(false), dtype)
                          : ib->MeanExt(x, axis, ib->Value<bool>(true), dtype);

    auto keepdim_opt = mindspore::GetScalarValue<bool>(keepdim->BuildValue());
    if (!keepdim_opt.has_value()) {
      grad = DynamicGradUnsqueeze(ib, x, axis, dout, keepdim, ib->Value<int64_t>(rank));
    } else if (!keepdim_opt.value() && rank > 1) {
      auto output_shape_keepdim = ib->ShapeCalc(std::make_shared<ReduceShapeShapeCalc>(false), {x, axis}, {1})[0];
      grad = ib->Reshape(dout, output_shape_keepdim);
    }

    grad = ib->Cast(grad, ib->GetDtype(dout));
    grad = dof ? ib->Mul(grad, dof) : ib->Muls(grad, ib->Value<float>(dof_imm));
    return ib->Mul(grad, ib->Sub(x, mean));
  }
}

NodePtr MinOrMaxGrad(Emitter *ib, const NodePtr &x, const NodePtr &axis, bool keep_dims, const NodePtr &out,
                     const NodePtr &dout) {
  auto y = out;
  auto grad = dout;
  if (!keep_dims) {
    auto output_shape_kept_dims = ib->ShapeCalc(std::make_shared<ReduceShapeShapeCalc>(), {x, axis}, {1})[0];
    y = ib->Reshape(out, output_shape_kept_dims);
    grad = ib->Reshape(dout, output_shape_kept_dims);
  }
  auto indicators = ib->Cast(ib->Equal(y, x), grad->dtype());
  auto num_selected = ib->SumExt(indicators, axis, ib->Value(true));
  return ib->Div(grad, num_selected) * indicators;
}

inline NodePtr ScatterZeroDim(Emitter *ib, const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                              const NodePtr &src, const NodePtr &reduce) {
  // Scatter op: ZeroDim need to expand to OneDim
  auto input_expand = ib->ExpandDims(input, -1);
  auto index_expand = ib->ExpandDims(index, -1);
  auto src_expand = ib->ExpandDims(src, -1);
  auto out = ib->Emit("TensorScatterElements", {input_expand, index_expand, src_expand, dim, reduce});
  // recover OneDim To ZeroDim
  return ib->Squeeze(out, MakeValue(ShapeVector{0}));
}

NodePtr Scatter_(BpropBuilder *ib, const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                 const NodePtr &reduce) {
  auto dim_val = dim->BuildValue();
  if (!IsValueKnown(dim_val)) {
    MS_EXCEPTION(ValueError) << "For Scatter, the `axis` must currently be a constant!";
  }
  auto input_shape = ib->GetShape(input);
  if (input_shape.size() == 0) {
    return ScatterZeroDim(ib, input, dim, index, src, reduce);
  } else if (IsDynamicRank(input_shape)) {
    auto rank = ib->Emit("Rank", {input});
    auto is_zero_dim_cond = ib->Emit("scalar_eq", {rank, ib->Value<int64_t>(0)});
    auto scatter_zero_dim_impl = [&input, &dim, &index, &src, &reduce](Emitter *e) -> NodePtrList {
      return {ScatterZeroDim(e, input, dim, index, src, reduce)};
    };
    auto scatter_impl = [&input, &dim, &index, &src](Emitter *e) -> NodePtrList {
      return {e->InplaceScatterSrc(input, dim, index, src)};
    };
    return ib->Conditional(is_zero_dim_cond, scatter_zero_dim_impl, scatter_impl);
  }
  return ib->InplaceScatterSrc(input, dim, index, src);
}

NodePtr ScatterOrTensorScatterElements(BpropBuilder *ib, const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                       const NodePtr &src, const NodePtr &reduce) {
  auto dim_val = dim->BuildValue();
  if (!IsValueKnown(dim_val)) {
    NodePtr dx_zeros = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
    (void)ib->InplaceScatterSrc(dx_zeros, dim, index, src);
    return dx_zeros;
  }
  auto input_shape = ib->GetShape(input);
  if (input_shape.size() == 0) {
    return ScatterZeroDim(ib, input, dim, index, src, reduce);
  } else if (IsDynamicRank(input_shape)) {
    auto rank = ib->Emit("Rank", {input});
    auto is_zero_dim_cond = ib->Emit("scalar_eq", {rank, ib->Value<int64_t>(0)});
    auto scatter_zero_dim_impl = [&input, &dim, &index, &src, &reduce](Emitter *e) -> NodePtrList {
      return {ScatterZeroDim(e, input, dim, index, src, reduce)};
    };
    auto scatter_impl = [&input, &dim, &index, &src](Emitter *e) -> NodePtrList {
      return {e->InplaceScatterSrc(input, dim, index, src)};
    };
    return ib->Conditional(is_zero_dim_cond, scatter_zero_dim_impl, scatter_impl);
  }
  return ib->InplaceScatterSrc(input, dim, index, src);
}

NodePtr ArgminOrArgmaxGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max, const bool is_minmax_dim) {
  auto keep_dims_value = keep_dims->BuildValue();
  size_t out_index = is_minmax_dim ? 0 : 1;
  size_t indices_index = is_minmax_dim ? 1 : 0;
  NodePtr dout_value = ib->TupleGetItem(dout, out_index);
  NodePtr indices = ib->TupleGetItem(out, indices_index);
  auto input_shape = ib->GetShape(x);
  if (IsValueKnown(keep_dims_value) && !IsDynamicRank(input_shape)) {
    auto is_zero_dim = input_shape.size() == 0;
    auto keep_dims_bool = GetValue<bool>(keep_dims_value);
    indices = (keep_dims_bool || is_zero_dim) ? indices : ib->ExpandDims(indices, axis);
    dout_value = (keep_dims_bool || is_zero_dim) ? dout_value : ib->ExpandDims(dout_value, axis);
  } else {
    auto rank = ib->Emit("Rank", {x});
    auto rank_is_zero = ib->Emit("scalar_eq", {rank, ib->Value<int64_t>(0)});
    auto cond = ib->LogicalOr(ib->ScalarToTensor(keep_dims, kBool), ib->ScalarToTensor(rank_is_zero, kBool));
    auto indices_expand = [&indices, &axis](Emitter *e) -> NodePtrList { return {e->ExpandDims(indices, axis)}; };
    auto indices_ori = [&indices](Emitter *e) -> NodePtrList { return {indices}; };
    indices = ib->Conditional(cond, indices_ori, indices_expand);
    auto dout_expand = [&dout_value, &axis](Emitter *e) -> NodePtrList { return {e->ExpandDims(dout_value, axis)}; };
    auto dout_ori = [&dout_value](Emitter *e) -> NodePtrList { return {dout_value}; };
    dout_value = ib->Conditional(cond, dout_ori, dout_expand);
  }
  NodePtr dx_zeros = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  auto reduce_value = ib->Value(static_cast<int64_t>(Reduce::REDUCE_NONE));
  auto dx = Scatter_(ib, dx_zeros, axis, indices, dout_value, reduce_value);
  return dx;
}

NodePtr MeidanDimGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                      const NodePtr &out, const NodePtr &dout) {
  return ReduceCommonOpGrad(ib, x, axis, keep_dims, out, dout, i0, i1);
}

inline NodePtr ReduceCommonOpGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                                  const NodePtr &out, const NodePtr &dout, int64_t dout_index, int64_t indices_index) {
  auto input_shape = ib->GetShape(x);
  NodePtr dout_value = ib->TupleGetItem(dout, dout_index);
  NodePtr indices = ib->TupleGetItem(out, indices_index);
  auto keep_dims_value = keep_dims->BuildValue();
  if (IsValueKnown(keep_dims_value) && !IsDynamicRank(input_shape)) {
    auto is_zero_dim = input_shape.size() == 0;
    auto keep_dims_bool = GetValue<bool>(keep_dims_value);
    indices = (keep_dims_bool || is_zero_dim) ? indices : ib->ExpandDims(indices, axis);
    dout_value = (keep_dims_bool || is_zero_dim) ? dout_value : ib->ExpandDims(dout_value, axis);
  } else {
    auto rank = ib->Emit("Rank", {x});
    auto rank_is_zero = ib->Emit("scalar_eq", {rank, ib->Value<int64_t>(0)});
    auto cond = ib->LogicalOr(ib->ScalarToTensor(keep_dims, kBool), ib->ScalarToTensor(rank_is_zero, kBool));
    auto indices_expand = [&indices, &axis](Emitter *e) -> NodePtrList { return {e->ExpandDims(indices, axis)}; };
    auto indices_ori = [&indices](Emitter *e) -> NodePtrList { return {indices}; };
    indices = ib->Conditional(cond, indices_ori, indices_expand);
    auto dout_expand = [&dout_value, &axis](Emitter *e) -> NodePtrList { return {e->ExpandDims(dout_value, axis)}; };
    auto dout_ori = [&dout_value](Emitter *e) -> NodePtrList { return {dout_value}; };
    dout_value = ib->Conditional(cond, dout_ori, dout_expand);
  }
  NodePtr dx_zeros = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  auto reduce_value = ib->Value(static_cast<int64_t>(Reduce::REDUCE_NONE));
  auto dx = ScatterOrTensorScatterElements(ib, dx_zeros, axis, indices, dout_value, reduce_value);
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

NodePtr LGamma(BpropBuilder *ib, const NodePtr &x) {
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
  auto log_sqrt_two_pi = ib->ValueByType((log_2 + log_pi) / 2, input_dtype);
  auto lanczos_gamma_plus_one_half = k_lanczos_gamma + 0.5;
  auto log_lanczos_gamma_plus_one_half = log(lanczos_gamma_plus_one_half);
  auto inf = std::numeric_limits<double>::infinity();
  auto infinity = ib->Fill(inf, ib->Shape(x), input_dtype->type_id());
  auto need_to_reflect = ib->Less(x, one_half);
  auto neg_input = ib->Neg(x);
  auto z = ib->Select(need_to_reflect, neg_input, ib->Sub(x, one));
  auto CalculateReflectedX = [&ib, &z, &k_base_lanczos_coeff, &k_lanczos_coefficients]() -> NodePtr {
    auto z_dtype = ib->GetDtype(z);
    NodePtr reflex_x = ib->ValueByType(k_base_lanczos_coeff, z_dtype);
    for (int i = 0; i < 8; ++i) {
      auto btmp = ib->AddScalar(z, ib->ValueByType(i, z_dtype), ib->Value(1));
      btmp = ib->AddScalar(btmp, (ib->ValueByType(1, z_dtype)), ib->Value(1));
      auto product = ib->RealDiv((ib->Tensor(k_lanczos_coefficients[i], z_dtype)), btmp);
      reflex_x = ib->AddScalar(product, reflex_x, ib->Value(1));
    }
    return reflex_x;
  };
  auto reflex_x = CalculateReflectedX();
  auto lanczos_tensor = ib->Tensor(lanczos_gamma_plus_one_half, input_dtype);
  auto log_lanczos_tensor = ib->ValueByType(log_lanczos_gamma_plus_one_half, input_dtype);
  auto t = ib->Add(z, lanczos_tensor);
  auto log_t = ib->AddScalar((ib->Log1p(ib->RealDiv(z, lanczos_tensor))), log_lanczos_tensor, ib->Value(1));
  auto log_y = ib->AddScalar(
    (ib->Add((ib->Log(reflex_x)), (ib->Mul((ib->Sub((ib->Add(z, one_half)), (ib->RealDiv(t, log_t)))), log_t)))),
    log_sqrt_two_pi, ib->Value(1));
  auto abs_input = ib->Abs(x);
  auto abs_frac_input = ib->Sub(abs_input, (ib->Floor(abs_input)));
  auto new_x = ib->Select(ib->LessEqual(x, zero), ib->Select(ib->Equal(abs_frac_input, zero), infinity, x), x);
  auto reduced_frac_input =
    ib->Select(ib->Greater(abs_frac_input, one_half), ib->Sub(one, abs_frac_input), abs_frac_input);
  auto reflection_denom =
    ib->Log(ib->Sin(ib->Muls(reduced_frac_input, ib->ValueByType(pi, ib->GetDtype(reduced_frac_input)))));
  auto reflection = ib->Select(ib->IsFinite(reflection_denom),
                               ib->AddScalar((ib->Sub((ib->Neg(reflection_denom)), log_y)),
                                             ib->ValueByType(log_pi, ib->GetDtype(log_y)), ib->Value(1)),
                               ib->Neg(reflection_denom));
  auto result = ib->Select(need_to_reflect, reflection, log_y);
  return ib->Select(ib->IsFinite(new_x), result, infinity);
}

bool CheckType(const TypePtr &check_type, const std::set<TypePtr> &template_types) {
  return std::any_of(template_types.begin(), template_types.end(), [&check_type](const TypePtr &accept) -> bool {
    return IsIdentidityOrSubclass(check_type, accept);
  });
}

ShapeVector PoolToNHWC(const ShapeVector &v) {
  ShapeVector new_v(v);
  new_v[i1] = v[i2];
  new_v[i2] = v[i3];
  new_v[i3] = v[i1];
  return new_v;
}

ShapeVector ConvToNHWC(const ShapeVector &v) {
  ShapeVector new_v(v);
  new_v[i0] = v[i1];
  new_v[i1] = v[i2];
  new_v[i2] = v[i3];
  new_v[i3] = 1;
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

NodePtr MatrixTranspose(BpropBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  if (IsDynamicRank(shape)) {
    auto dim = ib->Emit("Rank", {x});
    auto perm = ib->Range(dim);
    auto stridedslice_helper = [&perm, &ib](int64_t begin, int64_t end, int64_t step, int64_t end_mask = 0) {
      return ib->Emit("StridedSlice",
                      {perm, ib->Value<ShapeVector>(ShapeVector{begin}), ib->Value<ShapeVector>(ShapeVector{end}),
                       ib->Value<ShapeVector>(ShapeVector{step}), ib->Value<int64_t>(0LL), ib->Value<int64_t>(end_mask),
                       ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL)});
    };
    auto part_1 = stridedslice_helper(0, -2, 1);
    auto part_2 = stridedslice_helper(-1, 0, 1, 1);
    auto part_3 = stridedslice_helper(-2, -1, 1);
    perm = ib->Concat({part_1, part_2, part_3}, -1);
    return ib->Transpose(x, ib->TensorToTuple(perm));
  }
  auto dim = shape.size();
  if (dim < i2) {
    MS_LOG_EXCEPTION << "For MatrixTranspose, input's ndim " << dim << " is less or equal to 2, which is invalid";
  }
  std::vector<int64_t> perm(dim);
  for (size_t i = 0; i < dim; i++) {
    perm[i] = static_cast<int64_t>(i);
  }
  std::swap(perm[dim - i2], perm[dim - i1]);
  return ib->Transpose(x, perm);
}

NodePtr MatrixTransposeExt(BpropBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  if (IsDynamicRank(shape)) {
    auto dim = ib->Emit("Rank", {x});
    auto perm = ib->Range(dim);
    auto stridedslice_helper = [&perm, &ib](int64_t begin, int64_t end, int64_t step, int64_t end_mask = 0) {
      return ib->Emit("StridedSlice",
                      {perm, ib->Value<ShapeVector>(ShapeVector{begin}), ib->Value<ShapeVector>(ShapeVector{end}),
                       ib->Value<ShapeVector>(ShapeVector{step}), ib->Value<int64_t>(0LL), ib->Value<int64_t>(end_mask),
                       ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL)});
    };
    auto part_1 = stridedslice_helper(0, -2, 1);
    auto part_2 = stridedslice_helper(-1, 0, 1, 1);
    auto part_3 = stridedslice_helper(-2, -1, 1);
    perm = ib->Concat({part_1, part_2, part_3}, -1);
    return ib->Transpose(x, ib->TensorToTuple(perm));
  }
  auto dim = shape.size();
  if (dim < i2) {
    return x;
  }
  std::vector<int64_t> perm(dim);
  for (size_t i = 0; i < dim; i++) {
    perm[i] = static_cast<int64_t>(i);
  }
  std::swap(perm[dim - i2], perm[dim - i1]);
  return ib->Transpose(x, perm);
}

NodePtr Adjoint(BpropBuilder *ib, const NodePtr &x) { return MatrixTranspose(ib, ib->Conj(x)); }

NodePtr VectorNormGrad(BpropBuilder *ib, const NodePtr &input_node, const NodePtr &p, const NodePtr &dim_node,
                       const NodePtr &keepdim, const NodePtr &out_node, const NodePtr &dout_node) {
  auto dim = dim_node;
  auto dim_type = dim->abstract()->BuildType();
  if (dim_type->isa<TypeNone>()) {
    dim = ib->Value<std::vector<int64_t>>({});
  }
  auto keepdim_value = GetScalarValue<bool>(keepdim->BuildValue());
  auto input = input_node;
  float p_value = GetValue<float>(p->BuildValue());
  auto tensor_zero = ib->Tensor(0, input->dtype());
  auto input_shape = ib->GetShape(input);
  auto out = out_node;
  auto dout = dout_node;
  if (!keepdim_value.has_value()) {
    auto true_branch = [&](Emitter *e) -> NodePtrList {
      return {GetUnsqueezeTensor(e, input, dim, true, {out, dout})};
    };
    auto false_branch = [&](Emitter *e) -> NodePtrList {
      return {GetUnsqueezeTensor(e, input, dim, false, {out, dout})};
    };
    auto keepdim_true = ib->Equal(keepdim, ib->Value<bool>(true));
    auto outputs = ib->Conditional(keepdim_true, true_branch, false_branch);
    out = ib->TupleGetItem(outputs, 0);
    dout = ib->TupleGetItem(outputs, 1);
  } else {
    auto out_dout = GetUnsqueezeTensor(ib, input, dim, keepdim_value.value(), {out, dout});
    out = out_dout[0];
    dout = out_dout[1];
  }
  if (p_value == 0.0) {
    return ib->OutZeros(input);
  }
  if (p_value == 1.0) {
    return ib->Mul(dout, ib->Sign(input));
  }
  if (p_value == 2.0) {
    auto scale_v = ib->Div(input, out);
    auto equal_zero = ib->Equal(out, ib->Tensor(0, ib->GetDtype(out)));
    scale_v = ib->InplaceMaskedFillTensor(scale_v, equal_zero, ib->Tensor(0.0, ib->GetDtype(scale_v)));
    return ib->Mul(dout, scale_v);
  }
  if (std::isinf(p_value)) {
    auto input_abs = ib->Abs(input);
    auto input_sgn = ib->Sign(input);
    auto input_typeid = ib->GetDtypeId(input);
    // For Primitive 'IsNan', input's dtype cannot be bfloat16.
    if (input_typeid == kNumberTypeBFloat16) {
      input = ib->Cast(input, kFloat32);
      out = ib->Cast(out, kFloat32);
    }
    auto input_nan = ib->Emit("IsNan", {input});
    auto out_nan = ib->Emit("IsNan", {out});
    auto input_and_out_nan = ib->LogicalAnd(input_nan, out_nan);
    auto equal_max = ib->Cast(ib->LogicalOr(ib->Equal(input_abs, out), input_and_out_nan), input->dtype());
    auto input_scaled = ib->Mul(input_sgn, equal_max);
    auto max_cnt = ib->SumExt(ib->NotEqual(equal_max, tensor_zero), dim, ib->Value(true), ib->EmitValue(kNone));
    auto scale_v = ib->Div(dout, max_cnt);
    auto equal_zero = ib->Equal(out, ib->Tensor(0, ib->GetDtype(out)));
    scale_v = ib->InplaceMaskedFillTensor(scale_v, equal_zero, ib->Tensor(0.0, ib->GetDtype(scale_v)));
    auto grad_input = ib->Mul(input_scaled, scale_v);
    if (input_typeid == kNumberTypeBFloat16) {
      grad_input = ib->Cast(grad_input, kBFloat16);
    }
    return grad_input;
  }
  if (p_value < 1.0) {
    auto input_abs = ib->Abs(input);
    auto input_sgn = ib->Sign(input);
    auto input_pow = ib->PowTensorScalar(input_abs, ib->Value(p_value - 1));
    auto equal_zero = ib->Equal(input, ib->Tensor(0, ib->GetDtype(input)));
    auto input_fill = ib->InplaceMaskedFillTensor(input_pow, equal_zero, ib->Tensor(0.0, ib->GetDtype(input_pow)));
    auto input_scaled = ib->Mul(input_sgn, input_fill);
    auto out_pow = ib->PowTensorScalar(out, ib->Value(1 - p_value));
    return ib->Mul(ib->Mul(input_scaled, dout), out_pow);
  }
  if (p_value < 2.0) {
    auto input_abs = ib->Abs(input);
    auto input_sgn = ib->Sign(input);
    auto input_scaled = ib->Mul(ib->PowTensorScalar(input_abs, ib->Value(p_value - 1)), input_sgn);
    auto scale_v = ib->Div(dout, ib->PowTensorScalar(out, ib->Value(p_value - 1)));
    auto equal_zero = ib->Equal(out, ib->Tensor(0, ib->GetDtype(out)));
    scale_v = ib->InplaceMaskedFillTensor(scale_v, equal_zero, ib->Tensor(0.0, ib->GetDtype(scale_v)));
    return ib->Mul(input_scaled, scale_v);
  }
  auto input_abs = ib->Abs(input);
  auto input_scaled = ib->Mul(ib->PowTensorScalar(input_abs, ib->Value(p_value - 2)), input);
  auto scale_v = ib->Div(dout, ib->PowTensorScalar(out, ib->Value(p_value - 1)));
  auto equal_zero = ib->Equal(out, ib->Tensor(0, ib->GetDtype(out)));
  scale_v = ib->InplaceMaskedFillTensor(scale_v, equal_zero, ib->Tensor(0.0, ib->GetDtype(scale_v)));
  return ib->Mul(input_scaled, scale_v);
}
}  // namespace mindspore::expander::bprop
