/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <iterator>
#include <unordered_set>
#include <vector>
#include <cmath>

#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"
#include "ir/functor.h"
#include "include/op_enum.h"
#include "primitive/math_ops.h"
#include "ops_utils/op_utils.h"
#include "abstract/dshape.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore::expander::bprop {
namespace {
double PowFetchScalarValue(const ValuePtr &value_ptr, const std::string &arg_name, const std::string &op_name) {
  using value_func = std::function<double(const ValuePtr &value_ptr)>;
  const std::unordered_map<TypeId, value_func> func_map{{kNumberTypeBool,
                                                         [](const ValuePtr &value_ptr) -> double {
                                                           auto value_opt = mindspore::GetScalarValue<bool>(value_ptr);
                                                           return static_cast<double>(value_opt.value());
                                                         }},
                                                        {kNumberTypeInt64,
                                                         [](const ValuePtr &value_ptr) -> double {
                                                           auto value_opt =
                                                             mindspore::GetScalarValue<int64_t>(value_ptr);
                                                           return static_cast<double>(value_opt.value());
                                                         }},
                                                        {kNumberTypeFloat32, [](const ValuePtr &value_ptr) -> double {
                                                           auto fp32imm_ptr = value_ptr->cast<FP32ImmPtr>();
                                                           MS_EXCEPTION_IF_NULL(fp32imm_ptr);
                                                           return ops::GetDoubleValueFromScalar(fp32imm_ptr);
                                                         }}};

  auto type_id = value_ptr->type()->type_id();
  auto it = func_map.find(type_id);
  if (it == func_map.end()) {
    MS_LOG_EXCEPTION << "For " << op_name << ", got an invalid '" << arg_name << "' type: " << TypeIdToString(type_id);
  }
  return it->second(value_ptr);
}
}  // namespace
NodePtrList AddnGradFunc(BpropBuilder *ib) {
  auto dout = ib->GetInput(i2);
  auto x_abs = ib->GetInput(i0)->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  auto x_seq_ptr = x_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(x_seq_ptr);
  auto x_len = x_seq_ptr->elements().size();
  NodePtrList result(x_len, dout);
  if (x_abs->isa<abstract::AbstractList>()) {
    return {ib->MakeList(result)};
  }
  return {ib->MakeTuple(result)};
}

bool CloneInplaceInputFuncForInplaceDiv(const PynativeCallback &cb) {
  if (!cb.IsNotRequiresGrad(i1)) {
    return true;
  }
  return false;
}

void FreeTensorOfInplaceDivTensor(const PynativeCallback &cb) {
  auto &inputs = *cb.GetInputs();
  if (inputs[i1]->isa<tensor::Tensor>() && cb.IsNotRequiresGrad(i1)) {
    cb.FreeDeviceAddress(&inputs[0]);
    MS_LOG(DEBUG) << "Clear device address for inputs[0] of" << cb.opname();
  }
  cb.FreeOutputDeviceAddress();
}

NodePtrList IgammaBpropExpanderDyn(BpropBuilder *ib) {
  auto a = ib->GetInput(i0);
  auto x = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  auto sa = ib->Shape(a);
  auto sx = ib->Shape(x);
  auto rax = ib->BroadcastGradientArgs(sa, sx);
  auto ra = rax[0];
  auto rx = rax[1];
  auto partial_a = ib->Emit("IgammaGradA", {a, x});
  auto lgamma = LGamma(ib, a);
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)),
                     (ib->Mul((ib->SubScalar(a, ib->ValueByType(1, ib->GetDtype(a)), ib->Value(1))), (ib->Log(x)))))),
            lgamma));
  auto r1 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_a, dout), ra, false, true), sa);
  auto r2 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_x, dout), rx, false, true), sx);
  return {r1, r2};
}

NodePtrList IgammaBpropExpander(BpropBuilder *ib) {
  auto a = ib->GetInput(i0);
  auto x = ib->GetInput(i1);
  auto sa = ib->GetShape(a);
  auto sx = ib->GetShape(x);
  if (IsDynamic(sa) || IsDynamic(sx)) {
    return IgammaBpropExpanderDyn(ib);
  }
  auto rax = BroadcastGradientArgsInferValue(sa, sx);
  auto ra = rax[0];
  auto rx = rax[1];
  auto partial_a = ib->Emit("IgammaGradA", {a, x});
  auto lgamma = ib->Emit("Lgamma", {a});
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)),
                     (ib->Mul((ib->SubScalar(a, ib->ValueByType(1, ib->GetDtype(a)), ib->Value(1))), (ib->Log(x)))))),
            lgamma));
  auto dout = ib->GetInput(i3);
  NodePtr r1 = nullptr;
  NodePtr r2 = nullptr;
  if (!ra.empty()) {
    r1 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_a, dout), ra), sa);
  } else {
    r1 = ib->Reshape(ib->Mul(partial_a, dout), sa);
  }
  if (!rx.empty()) {
    r2 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_x, dout), rx), sx);
  } else {
    r2 = ib->Reshape(ib->Mul(partial_x, dout), sx);
  }
  return {r1, r2};
}

inline NodePtr SelectScalar(BpropBuilder *ib, const NodePtr &cond, const NodePtr &x_scalar, const NodePtr &y,
                            const NodePtr &input0, const NodePtr &input1) {
  if (ib->IsGraphMode() && !(IsDynamic(ib->GetShape(input0)) || IsDynamic(ib->GetShape(input1)))) {
    // Notice: This is just a temporary evasion! In order to avoid a fusion problem of the
    // MaskedFill operator in the static shape of graph mode.
    // this code will be deleted after the problem is fixed.
    auto x = ib->Emit("FillV2", {ib->Shape(y), x_scalar});
    return ib->Select(cond, x, y);
  } else {
    return ib->MaskedFill(y, cond, x_scalar);
  }
}

NodePtr MaybeMultiply(BpropBuilder *ib, const TypePtr &input_type, const NodePtr &t, const NodePtr &s,
                      const std::string &arg_name) {
  bool is_one = false;
  auto s_ptr = s->BuildValue();
  MS_EXCEPTION_IF_NULL(s_ptr);
  if (!s_ptr->isa<ValueAny>()) {
    auto s_type = ib->GetDtypeId(s);
    if (s_type == kNumberTypeInt64) {
      auto s_value = GetValue<int64_t>(s_ptr);
      is_one = s_value == 1;
    } else if (s_type == kNumberTypeFloat32) {
      auto s_value = GetValue<float>(s_ptr);
      is_one = s_value == 1.0;
    } else if (s_type == kNumberTypeBool) {
      auto s_value = GetValue<bool>(s_ptr);
      is_one = s_value == True;
    } else {
      MS_EXCEPTION(TypeError) << "For " << ib->name() << " grad " << arg_name << "'s type is wrong!";
    }
  }
  auto out = is_one ? t : ib->Muls(t, s);
  return out;
}

static bool IsColMajor2DTensor(const NodePtr &node, const std::string &op_name) {
  auto val = node->BuildValue();
  if (val == nullptr) {
    return false;
  }
  auto tensor = val->cast<tensor::TensorPtr>();
  // can not cast to tensor in graph mode
  if (tensor == nullptr) {
    return false;
  }
  auto info = tensor->storage_info();
  // can not obtain storage_info if current tensor is not view operator's output
  if (info == nullptr) {
    return false;
  }
  const auto &shape = info->shape;
  const auto &strides = info->strides;
  if (shape.size() != 2 || strides.size() != 2) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " grad, got an invalid shape or strides of input tensor: " << shape
                             << " or " << strides;
  }
  return (strides[0] == 1 && strides[1] == shape[0]);
}

static NodePtr MmMat1Backward(BpropBuilder *ib, const TypePtr &input_type, const NodePtr &dout, const NodePtr &mat1,
                              const NodePtr &mat2, const NodePtr &alpha) {
  // Mat1 Column-major optimization
  if (IsColMajor2DTensor(mat1, ib->name())) {
    auto alt = ib->Transpose(ib->MatMul(mat2, dout, false, true), {1, 0});
    return MaybeMultiply(ib, input_type, alt, alpha, "alpha");
  }
  return MaybeMultiply(ib, input_type, ib->MatMul(dout, mat2, false, true), alpha, "alpha");
}

static NodePtr MmMat2Backward(BpropBuilder *ib, const TypePtr &input_type, const NodePtr &dout, const NodePtr &mat1,
                              const NodePtr &mat2, const NodePtr &alpha) {
  // Mat2 Column-major optimization
  if (IsColMajor2DTensor(mat2, ib->name())) {
    auto alt = ib->Transpose(ib->MatMul(dout, mat1, true, false), {1, 0});
    return MaybeMultiply(ib, input_type, alt, alpha, "alpha");
  }
  return MaybeMultiply(ib, input_type, ib->MatMul(mat1, dout, true, false), alpha, "alpha");
}

// Encapsulate inverse-broadcast reduction and reshape for input gradient.
inline NodePtr BroadcastReduceToInputShape(BpropBuilder *ib, const NodePtr &input, const NodePtr &dout,
                                           const NodePtr &input_grad_in) {
  auto input_shape = ib->GetShape(input);
  auto dout_shape = ib->GetShape(dout);
  NodePtr input_grad = input_grad_in;
  if (IsDynamic(input_shape) || IsDynamic(dout_shape)) {
    auto bc_axis = ib->BroadcastGradientArgs(input, dout);
    auto true_branch = [input_grad](Emitter *e) -> NodePtrList { return {input_grad}; };
    auto false_branch = [&input_grad, &bc_axis, &input](Emitter *e) -> NodePtrList {
      return {e->Reshape(e->SumExt(input_grad, bc_axis[i0], e->Value(false), e->EmitValue(kNone)), e->Shape(input))};
    };
    auto cond = ib->Equal(ib->Emit("sequence_len", {bc_axis[i0]}), ib->Value<int64_t>(0));
    return ib->Conditional(cond, true_branch, false_branch);
  }
  auto bc_axis = BroadcastGradientArgsInferValue(input_shape, dout_shape);
  if (!bc_axis[i0].empty()) {
    input_grad = ib->SumExt(input_grad, ib->Value(bc_axis[i0]), ib->Value(false), ib->EmitValue(kNone));
  }
  if (ib->GetRank(input_grad) != input_shape.size()) {
    input_grad = ib->Reshape(input_grad, input_shape);
  }
  return input_grad;
}

NodePtrList MinimumMaximumGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dout,
                               bool is_minimum) {
  NodePtr grad_x = nullptr;
  NodePtr grad_y = nullptr;
  if (!x->need_compute_grad_out() && !y->need_compute_grad_out()) {
    return {grad_x, grad_y};
  }
  auto half_dout = ib->Cast(ib->Divs(dout, ib->ValueByType(2, ib->GetDtype(dout))), ib->GetDtype(x));
  auto equal_mask = ib->Equal(x, y);
  auto zeros = ib->Tensor(0, ib->GetDtype(dout));
  auto is_less = ib->Less(x, y);
  auto is_greater = ib->Greater(x, y);
  auto dout_sel = ib->Select(equal_mask, half_dout, dout);
  if (x->need_compute_grad_out()) {
    if (is_minimum) {
      grad_x = SelectScalar(ib, is_greater, zeros, dout_sel, x, y);
    } else {
      grad_x = SelectScalar(ib, is_less, zeros, dout_sel, x, y);
    }
  }
  if (y->need_compute_grad_out()) {
    if (is_minimum) {
      grad_y = SelectScalar(ib, is_less, zeros, dout_sel, x, y);
    } else {
      grad_y = SelectScalar(ib, is_greater, zeros, dout_sel, x, y);
    }
  }
  return BinopGradCommon(ib, x, y, grad_x, grad_y);
}

NodePtrList CumMaxMinGrad(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto axis = ib->GetInput(i1);
  auto out = ib->GetInput(i2);
  auto indices = ib->TupleGetItem(out, i1);

  auto dout = ib->GetInput(i3);
  auto dout0 = ib->TupleGetItem(dout, i0);
  auto out_grad = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  (void)ib->InplaceScatterAdd(out_grad, axis, indices, dout0);
  return {out_grad, ib->OutZeros(axis)};
}

NodePtrList IndexAddGrad(BpropBuilder *ib) {
  auto dim = ib->GetInput(i1);
  auto index = ib->GetInput(i2);
  auto source = ib->GetInput(i3);
  auto alpha = ib->GetInput(i4);
  auto dout = ib->GetInput(i6);
  if (source->need_compute_grad_out()) {
    NodePtr dsource;
    if (ib->GetShape(source).size() > 0) {
      dsource = ib->ExpandAs(ib->IndexSelect(dout, dim, index), source);
    } else {
      ShapeVector axis{0};
      dsource = ib->IndexSelect(dout, dim, ib->Squeeze(index, MakeValue(axis)));
    }
    dsource = MaybeMultiply(ib, ib->GetDtype(source), dsource, alpha, "alpha");
    return {dout, ib->OutZeros(dim), ib->OutZeros(index), dsource, ib->OutZeros(alpha)};
  }
  return {dout, ib->OutZeros(dim), ib->OutZeros(index), ib->OutZeros(source), ib->OutZeros(alpha)};
}

ShapeArray MatrixDeterminantShapeFunc(const ShapeArray &inputs) {
  auto new_shape = inputs.at(0);
  new_shape.push_back(1);
  new_shape.push_back(1);
  return {new_shape};
}

ShapeVector MatrixDeterminantInferFunc(const ShapeArray &inputs, const HashSet<size_t> &) {
  auto new_shape = inputs.at(0);
  return {IsDynamicRank(new_shape) ? -1 : SizeToLong(new_shape.size()) + 2};
}

NodePtrList BpropAddcCommon(BpropBuilder *ib, const std::string &op_name) {
  auto input_data = ib->GetInput(i0);
  auto x1 = ib->GetInput(i1);
  auto x2 = ib->GetInput(i2);
  auto value = ib->GetInput(i3);
  auto dout = ib->GetInput(i5);
  auto dinput_data = dout;
  auto dout_typeptr = ib->GetDtype(dout);

  input_data = ib->Cast(input_data, kFloat32);
  x1 = ib->Cast(x1, kFloat32);
  x2 = ib->Cast(x2, kFloat32);
  value = ib->Cast(value, kFloat32);
  if (op_name == "Addcdiv") {
    dinput_data = ib->Cast(dinput_data, kFloat32);
  }

  NodePtr inner_out = nullptr;
  NodePtr dx1 = nullptr;
  NodePtr dx2 = nullptr;
  NodePtr dvalue = nullptr;
  if (op_name == "Addcdiv") {
    constexpr int64_t const_val = -2;
    inner_out = ib->Add((ib->Mul(value, ib->Div(x1, x2))), input_data);
    dx2 = ib->Neg(ib->Mul(
      ib->Mul(ib->Mul(x1, value), ib->PowTensorScalar(x2, ib->ValueByType(const_val, ib->GetDtype(x2)))), dinput_data));
    dx1 = ib->Mul(dinput_data, ib->Div(value, x2));
    dvalue = ib->Mul(dinput_data, ib->Div(x1, x2));
  } else {
    dx1 = ib->Mul(dout, ib->Mul(value, x2));
    dx2 = ib->Mul(dout, ib->Mul(value, x1));
    inner_out = ib->Add((ib->Mul((ib->Mul(x1, x2)), value)), input_data);
    dvalue = ib->Mul(dout, ib->Mul(x1, x2));
  }
  auto tmp_dinput_data = BinopGradCommon(ib, inner_out, input_data, dout, dinput_data);
  dinput_data = tmp_dinput_data[1];
  auto tmp_dx1 = BinopGradCommon(ib, inner_out, x1, dout, dx1);
  dx1 = tmp_dx1[1];
  auto tmp_dx2 = BinopGradCommon(ib, inner_out, x2, dout, dx2);
  dx2 = tmp_dx2[1];
  auto tmp_dvalue = BinopGradCommon(ib, inner_out, value, dout, dvalue);
  dvalue = tmp_dvalue[1];

  dinput_data = ib->Cast(dinput_data, ib->GetDtype(ib->GetInput(i0)));
  dx1 = ib->Cast(dx1, ib->GetDtype(ib->GetInput(i1)));
  dx2 = ib->Cast(dx2, ib->GetDtype(ib->GetInput(i2)));
  dvalue = ib->Cast(dvalue, ib->GetDtype(ib->GetInput(i3)));

  return {dinput_data, dx1, dx2, dvalue};
}

std::optional<float> GetAlpha(const NodePtr &alpha) {
  auto alpha_value = alpha->BuildValue();
  if (alpha_value->isa<Int64Imm>()) {
    auto imm_int64_ptr = alpha_value->cast_ptr<Int64Imm>();
    MS_EXCEPTION_IF_NULL(imm_int64_ptr);
    auto imm_int64 = imm_int64_ptr->value();
    return std::make_optional(imm_int64);
  } else if (alpha_value->isa<FP32Imm>()) {
    auto imm_fp32_ptr = alpha_value->cast_ptr<FP32Imm>();
    MS_EXCEPTION_IF_NULL(imm_fp32_ptr);
    auto imm_fp32 = imm_fp32_ptr->value();
    return std::make_optional(imm_fp32);
  }

  return std::nullopt;
}

ShapeArray ReduceStdShapeFunc(const ShapeVector &x_shape, const ShapeVector &axis) {
  ShapeVector new_axis = axis;
  if (new_axis.empty() && !x_shape.empty()) {
    new_axis.reserve(x_shape.size());
    for (int64_t i = 0; i < SizeToLong(x_shape.size()); i++) {
      new_axis.push_back(i);
    }
  }
  (void)std::transform(new_axis.begin(), new_axis.end(), new_axis.begin(), [&x_shape](const int64_t &c) {
    if (c < 0) {
      return c + SizeToLong(x_shape.size());
    }
    return c;
  });
  for (size_t i = 1; i < new_axis.size(); ++i) {
    for (size_t j = 0; j < new_axis.size() - i; ++j) {
      if (new_axis[j] > (new_axis[j + 1])) {
        std::swap(new_axis[j], new_axis[j + 1]);
      }
    }
  }
  // input_x:[2,3,4,5]  new_axis:   [0, 2]
  // reduce: [3,5]      reshape:[1,3,1,5]
  auto reshape = x_shape;
  for (auto &i : new_axis) {
    reshape[LongToSize(i)] = 1;
  }
  int64_t num = 1;
  for (const auto &i : new_axis) {
    num *= x_shape[LongToSize(i)];
  }
  return {reshape, {num - 1}, {num}};
}

class ReduceStdShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_ReduceStd", ReduceStdShapeCalc)

  explicit ReduceStdShapeCalc(const std::vector<int64_t> &axis)
      : ShapeCalcFunctor("ShapeCalc_ReduceStd"), axis_(axis) {}

  ValuePtr ToValue() const override { return MakeValue(axis_); }

  void FromValue(const ValuePtr &value) override { axis_ = GetValue<std::vector<int64_t>>(value); }

  ShapeArray Calc(const ShapeArray &inputs) const override { return ReduceStdShapeFunc(inputs.at(0), axis_); }

  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : SizeToLong(shape_x.size());
    return {rank, 1, 1};
  }

 protected:
  std::vector<int64_t> axis_;
};

REG_FUNCTOR("ShapeCalc_ReduceStd", ReduceStdShapeCalc);

NodePtrList FminFmaxGrad(BpropBuilder *ib, bool if_fmin) {
  auto x1 = ib->GetInput(i0);
  auto x2 = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  auto x1_dtype = ib->GetDtype(x1);
  auto x2_dtype = ib->GetDtype(x2);
  x1 = ib->Cast(x1, kFloat32);
  x2 = ib->Cast(x2, kFloat32);
  dout = ib->Cast(dout, kFloat32);
  auto x1_nan = ib->Emit("IsNan", {x1});
  auto x2_nan = ib->Emit("IsNan", {x2});
  NodePtr b1 = nullptr;
  NodePtr b2 = nullptr;
  if (if_fmin) {
    b1 = ib->LogicalOr(ib->LessEqual(x1, x2), x2_nan);
    b2 = ib->LogicalOr(ib->Less(x2, x1), ib->LogicalAnd(x1_nan, ib->LogicalNot(x2_nan)));
  } else {
    b1 = ib->LogicalOr(ib->LogicalAnd(ib->GreaterEqual(x1, x2), ib->LogicalNot(x1_nan)), x2_nan);
    b2 = ib->LogicalOr(ib->LogicalAnd(ib->Greater(x2, x1), ib->LogicalNot(x2_nan)),
                       ib->LogicalAnd(x1_nan, ib->LogicalNot(x2_nan)));
  }
  auto rx1 = ib->MaskedFill(x1, b1, ib->Tensor(1.0, kFloat32));
  rx1 = ib->MaskedFill(rx1, ib->LogicalNot(b1), ib->Tensor(0.0, kFloat32));
  auto rx2 = ib->MaskedFill(x2, b2, ib->Tensor(1.0, kFloat32));
  rx2 = ib->MaskedFill(rx2, ib->LogicalNot(b2), ib->Tensor(0.0, kFloat32));
  auto rrx1 = ib->Mul(rx1, dout);
  auto rrx2 = ib->Mul(rx2, dout);
  auto shape_of_x1 = ib->Shape(x1);
  auto shape_of_x2 = ib->Shape(x2);
  auto x1_dim = ib->GetRank(x1);
  auto x2_dim = ib->GetRank(x2);
  NodePtr sum_r1;
  NodePtr sum_r2;
  if (x1_dim == 0 && x2_dim != 0) {
    sum_r1 = ib->ReduceSum(rrx1);
    sum_r2 = rrx2;
  } else if (x1_dim == 0 && x2_dim == 0) {
    sum_r1 = ib->ReduceSum(rrx1);
    sum_r2 = ib->ReduceSum(rrx2);
  } else if (x1_dim != 0 && x2_dim == 0) {
    sum_r2 = ib->ReduceSum(rrx2);
    sum_r1 = rrx1;
  } else {
    auto tmp = ib->BroadcastGradientArgs(x1, x2);
    auto rx = tmp[0];
    auto ry = tmp[1];
    sum_r1 = ib->ReduceSum(rrx1, rx, false, true);
    sum_r2 = ib->ReduceSum(rrx2, ry, false, true);
  }
  auto brrx1 = ib->Reshape(sum_r1, shape_of_x1);
  auto brrx2 = ib->Reshape(sum_r2, shape_of_x2);
  brrx1 = ib->Cast(brrx1, x1_dtype);
  brrx2 = ib->Cast(brrx2, x2_dtype);
  return {brrx1, brrx2};
}

NodePtr StdGrad(BpropBuilder *ib, const NodePtr &input, const NodePtr &axis, const NodePtr &correction,
                const NodePtr &keep_dims, const NodePtr &out, const NodePtr &dout) {
  auto grad_var = ib->Div(dout, ib->Muls(out, ib->Value<float>(2.0)));
  auto equal_zero = ib->Equal(out, ib->Tensor(0, ib->GetDtype(out)));
  grad_var = ib->MaskedFill(grad_var, equal_zero, ib->Tensor(0.0, ib->GetDtype(grad_var)));

  auto grad = VarGrad(ib, input, axis, grad_var, correction, keep_dims);
  return grad;
}

NodePtr MeanExtGrad(BpropBuilder *ib, const NodePtr &input, NodePtr axis, const NodePtr &keep_dims, const NodePtr &out,
                    const NodePtr &dout) {
  auto input_dtype_id = ib->GetDtypeId(input);
  if (input_dtype_id == kNumberTypeComplex64 || input_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'MeanExt', gradient not support for complex type currently.";
  }

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  NodePtr grad;
  auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
  if (!keep_dims_opt.has_value()) {
    auto true_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, true)}; };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, false)}; };
    auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
    grad = ib->Conditional(keep_dims_true, true_branch, false_branch);
  } else {
    grad = SumGrad(ib, input, axis, dout, keep_dims_opt.value());
  }
  grad = ib->Cast(grad, ib->GetDtype(input));

  NodePtr div_shape_node;
  NodePtr dx;
  if (IsDynamic(ib->GetShape(input)) || IsDynamic(ib->GetShape(out))) {
    auto shape_out_sz = ib->DynSize(out, kFloat32);
    auto true_branch = [&](Emitter *e) -> NodePtrList { return {ib->Tensor(1, kFloat32)}; };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {ib->DynSize(input, kFloat32) / shape_out_sz}; };
    auto is_zero_out_sz = ib->Equal(shape_out_sz, ib->Tensor(0, kFloat32));
    auto div_shape = ib->Conditional(is_zero_out_sz, true_branch, false_branch);
    div_shape_node = ib->Cast(div_shape, ib->GetDtype(grad));
    dx = ib->Div(grad, div_shape_node);
  } else {
    auto shape_out_sz = ib->GetSize(out);
    auto div_shape = shape_out_sz == 0 ? 1 : ib->GetSize(input) / shape_out_sz;
    div_shape_node = ib->ValueByType(div_shape, ib->GetDtype(grad));
    dx = ib->Divs(grad, div_shape_node);
  }
  return dx;
}

DEF_PURE_SHAPE_CALC(g_addbmm_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto batch1_shape = inputs.at(i0);
    auto batch2_shape = inputs.at(i1);
    ShapeVector ret_shape(batch1_shape.begin(), batch1_shape.end());
    ret_shape.back() = *(batch2_shape.end() - 1);
    return {ret_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto x_shape = inputs[0];
    auto rank = IsDynamicRank(x_shape) ? -1 : SizeToLong(x_shape.size());
    return {rank};
  });

DEF_PURE_SHAPE_CALC(g_fft_dct_axis2tuple)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    constexpr int64_t input_num = 2;
    if (inputs.size() != input_num) {
      MS_LOG_EXCEPTION << "ShapeCalc[g_fft_dct_axis2tuple] expect 2 inputs, but got " << inputs.size() << "inputs";
    }
    auto x_shape = inputs.at(i0);
    auto axis = inputs.at(i1)[0];

    axis = NormalizeAxis(axis, x_shape.size());
    std::vector<int64_t> res{axis};
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

DEF_PURE_SHAPE_CALC(g_fft_dct_axes)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shp = inputs.at(i0);
    auto s_v = inputs.at(i1);

    std::vector<int64_t> res;
    for (size_t i = 0; i < s_v.size(); i++) {
      (void)res.emplace_back(SizeToLong(x_shp.size() - s_v.size() + i));
    }
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto s_shape = inputs.at(i1);
    return {IsDynamicRank(s_shape) ? -1 : SizeToLong(s_shape.size())};
  });

DEF_PURE_SHAPE_CALC(g_fft_shape_dim)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    constexpr int64_t input_num = 1;
    if (inputs.size() != input_num) {
      MS_LOG_EXCEPTION << "ShapeCalc[g_fft_shape_dim] expect 1 inputs, but got " << inputs.size() << "inputs";
    }
    auto x_shape = inputs.at(i0);

    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_size;
    for (size_t i = 0; i < x_shape.size(); i++) {
      input_shape.emplace_back(x_shape[i]);
      input_size.emplace_back(i);
    }
    return {input_shape, input_size};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto x_shape = inputs.at(0);
    auto rank = IsDynamicRank(x_shape) ? -1 : SizeToLong(x_shape.size());
    return {rank, rank};
  });

DEF_PURE_SHAPE_CALC(g_fft_axes_shape)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shp = inputs.at(i0);
    auto axes_v = inputs.at(i1);

    int64_t axis;
    std::vector<int64_t> res;

    for (size_t i = 0; i < axes_v.size(); i++) {
      axis = NormalizeAxis(axes_v[i], x_shp.size());
      res.emplace_back(x_shp[LongToSize(axis)]);
    }
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto axes_shape = inputs.at(i1);
    return {IsDynamicRank(axes_shape) ? -1 : SizeToLong(axes_shape.size())};
  });

DEF_PURE_SHAPE_CALC(g_fft_cmpt_double_n_dim)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto dout_shp = inputs.at(i0);
    auto axes_v = inputs.at(i1);

    std::vector<int64_t> res;

    auto last_dim = NormalizeAxis(axes_v.back(), dout_shp.size());
    res.push_back(dout_shp[last_dim]);
    res.push_back(last_dim);
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {2}; });

NodePtr FFTNormReverse(BpropBuilder *ib, NodePtr norm) {
  constexpr int64_t kNormBackward = 0;
  constexpr int64_t kNormForward = 1;
  constexpr int64_t kNormOrtho = 2;

  // step1：Get the inputs needed to solve for the gradient.
  auto norm_type = norm->abstract()->BuildType();
  int64_t grad_norm_value = kNormForward;
  if (!norm_type->isa<TypeNone>()) {
    auto norm_value = GetValue<int64_t>(norm->BuildValue());
    switch (norm_value) {
      case kNormBackward:
        grad_norm_value = kNormForward;
        break;
      case kNormForward:
        grad_norm_value = kNormBackward;
        break;
      case kNormOrtho:
        grad_norm_value = kNormOrtho;
        break;
      default:
        break;
    }
  }
  return ib->Value(grad_norm_value);
}

NodePtrList FFTGradCommon(BpropBuilder *ib, const std::string &op_name) {
  auto x = ib->GetInput(i0);
  auto n = ib->GetInput(i1);
  auto dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  auto dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);

  // step2：Get the gradient.
  auto grad_dout = ib->Emit(op_name, {dout, n, dim, norm});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto n_type = n->abstract()->BuildType();
  if (!n_type->isa<TypeNone>()) {
    auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
    grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});
  }

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
}

inline NodePtr GradDiagonal(Emitter *ib, const NodePtr &dout, const NodePtr &dx_trans_shape,
                            std::tuple<int64_t, int64_t, int64_t, size_t> int_tuple, const TypePtr &x_dtype) {
  auto [offset, dim1, dim2, x_dim] = int_tuple;
  auto value = ib->Tensor(0, x_dtype);
  auto dx = ib->Emit("FillV2", {dx_trans_shape, value});
  auto k = ib->Tensor(offset, kInt32);
  constexpr int64_t max_length = 200000000;
  dx = ib->Emit("MatrixSetDiagV3", {dx, dout, k},
                {{"align", MakeValue("LEFT_RIGHT")}, {"max_length", MakeValue(max_length)}});
  int64_t dim = 0;
  ShapeVector perm(x_dim, 0);
  for (size_t i = 0; i < x_dim; ++i) {
    if (i == static_cast<size_t>(dim1)) {
      perm[i] = static_cast<int64_t>(x_dim - i2);
    } else if (i == static_cast<size_t>(dim2)) {
      perm[i] = static_cast<int64_t>(x_dim - i1);
    } else {
      perm[i] = dim;
      dim++;
    }
  }
  dx = ib->Transpose(dx, perm);
  return dx;
}

inline NodePtr GradDiagonalScalarToTensor(Emitter *ib, const NodePtr &dout, const NodePtr &dx_trans_shape,
                                          const NodePtr &diagonal, std::tuple<int64_t, int64_t, size_t> int_tuple,
                                          const TypePtr &x_dtype) {
  auto [dim1, dim2, x_dim] = int_tuple;
  auto value = ib->Tensor(0, x_dtype);
  auto dx = ib->Emit("FillV2", {dx_trans_shape, value});
  auto k = ib->ScalarToTensor(diagonal, kInt32);
  constexpr int64_t max_length = 200000000;
  dx = ib->Emit("MatrixSetDiagV3", {dx, dout, k},
                {{"align", MakeValue("LEFT_RIGHT")}, {"max_length", MakeValue(max_length)}});
  int64_t dim = 0;
  ShapeVector perm(x_dim, 0);
  for (size_t i = 0; i < x_dim; ++i) {
    if (i == static_cast<size_t>(dim1)) {
      perm[i] = static_cast<int64_t>(x_dim - i2);
    } else if (i == static_cast<size_t>(dim2)) {
      perm[i] = static_cast<int64_t>(x_dim - i1);
    } else {
      perm[i] = dim;
      dim++;
    }
  }
  dx = ib->Transpose(dx, perm);
  return dx;
}

inline NodePtr ReduceExtOpGetMask(BpropBuilder *ib, const NodePtr &x, const NodePtr &out) {
  auto out_is_nan = ib->IsNanFunc(out);
  auto input_is_nan = [&x](Emitter *e) -> NodePtrList { return {e->IsNanFunc(x)}; };
  auto input_equal_out = [&x, &out](Emitter *e) -> NodePtrList { return {e->Equal(x, out)}; };
  return ib->Conditional(out_is_nan, input_is_nan, input_equal_out);
}

inline NodePtr ReduceExtOpGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &out, const NodePtr &dout) {
  auto mask = ReduceExtOpGetMask(ib, x, out);
  auto out_grad = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  auto mask_sum = ib->SumExt(mask, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone));
  auto grad_div_mask_sum = ib->Div(dout, ib->Cast(mask_sum, ib->GetDtype(dout)));
  grad_div_mask_sum = ib->Reshape(ib->Cast(grad_div_mask_sum, ib->GetDtype(x)), ShapeVector{});
  auto dx = ib->InplaceMaskedFillTensor(out_grad, mask, grad_div_mask_sum);
  return {dx};
}

class DiagonalShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_Diagonal", DiagonalShapeCalc)

  explicit DiagonalShapeCalc(int64_t dim1, int64_t dim2)
      : ShapeCalcFunctor("ShapeCalc_Diagonal"), dim1_(dim1), dim2_(dim2) {}

  ValuePtr ToValue() const override {
    auto values = {MakeValue(dim1_), MakeValue(dim2_)};
    return std::make_shared<ValueTuple>(values);
  }

  void FromValue(const ValuePtr &value) override {
    auto values = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(values);
    if (values->value().size() != i2) {
      MS_LOG(EXCEPTION) << "DiagonalShapeCalc's value size should be 2, but got " << values->value().size();
    }
    dim1_ = GetValue<int64_t>(values->value()[0]);
    dim2_ = GetValue<int64_t>(values->value()[1]);
  }

  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs.at(i0);
    auto out_shape = inputs.at(i1);
    ShapeVector dx_trans_shape(out_shape.begin(), out_shape.end() - i1);
    dx_trans_shape.push_back(x_shape[static_cast<size_t>(dim1_)]);
    dx_trans_shape.push_back(x_shape[static_cast<size_t>(dim2_)]);
    return {dx_trans_shape};
  }

  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    auto out_shape = inputs.at(1);
    auto rank = IsDynamicRank(out_shape) ? -1 : SizeToLong(out_shape.size() + 1);
    return {rank};
  }

 protected:
  int64_t dim1_;
  int64_t dim2_;
};

REG_FUNCTOR("ShapeCalc_Diagonal", DiagonalShapeCalc);

void FreeTensorsOfMul(const PynativeCallback &cb) {
  cb.FreeOutputDeviceAddress();
  // For operators like Mul, the dx ONLY rely on y, and dy ONLY rely on x.
  // so if y is a valuenode, the dy is useless, we can free x in ahead.
  auto &inputs = *cb.GetInputs();
  if (cb.IsNotRequiresGrad(i0) && inputs[i1]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i1]);
    MS_LOG(DEBUG) << "Clear device address for inputs[1] of " << cb.opname();
  }
  if (cb.IsNotRequiresGrad(i1) && inputs[i0]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i0]);
    MS_LOG(DEBUG) << "Clear device address for inputs[0] of " << cb.opname();
  }
}

void FreeTensorsOfBaddbmm(const PynativeCallback &cb) {
  cb.FreeOutputDeviceAddress();
  auto &inputs = *cb.GetInputs();
  cb.FreeDeviceAddress(&inputs[i0]);
  if (cb.IsNotRequiresGrad(i1) && inputs[i2]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i2]);
    MS_LOG(DEBUG) << "Clear device address for inputs[2] of " << cb.opname();
  }
  if (cb.IsNotRequiresGrad(i2) && inputs[i1]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i1]);
    MS_LOG(DEBUG) << "Clear device address for inputs[1] of " << cb.opname();
  }
}

void FreeTensorsOfDiv(const PynativeCallback &cb) {
  cb.FreeInputDeviceAddress({i0});
  // For operators like Div, the dy does not rely on output node, so if y is a valuenode, we can free output.
  if (cb.IsNotRequiresGrad(i1)) {
    cb.FreeOutputDeviceAddress();
  }
}

size_t FetchNumOfSequenceTensor(const NodePtr &input, const std::string &op_name, const std::string &arg_name) {
  auto abs = input->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (is_dyn_seq) {
    MS_LOG(EXCEPTION) << "For " << op_name << "'s backward, the tensor num in '" << arg_name
                      << "' should not be dynamic, which is not supported.";
  }
  auto sequence_shape = base_shape->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  auto num = sequence_shape->size();
  return num;
}

void CheckValueOfInt64Scalar(const NodePtr &node, const std::string &op_name, const std::string &arg_name,
                             int64_t expect_value) {
  auto value_opt = mindspore::GetScalarValue<int64_t>(node->BuildValue());
  if (!value_opt.has_value()) {
    MS_LOG(EXCEPTION) << "For " << op_name << "'s backward, '" << arg_name << "' should be const.";
  }
  auto real_value = value_opt.value();
  if (real_value != expect_value) {
    MS_LOG(EXCEPTION) << "For " << op_name << "'s backward, '" << arg_name << "' only support " << expect_value
                      << ", but got " << real_value;
  }
}

inline NodePtr ForEachTransposeLastTwoDim(BpropBuilder *ib, const NodePtr &node, size_t num) {
  std::vector<NodePtr> new_tensors;
  for (size_t i = 0; i < num; ++i) {
    auto tensor_i = ib->TupleGetItem(node, i);
    auto tensor_i_t = ib->Transpose(tensor_i, -1, -2);
    new_tensors.push_back(tensor_i_t);
  }
  return ib->MakeTuple(new_tensors);
}

inline NodePtr ForEachReshapeAs(BpropBuilder *ib, const NodePtr &inputs, const NodePtr &targets, size_t num) {
  std::vector<NodePtr> nodes;
  for (size_t i = 0; i < num; i++) {
    auto target_i = ib->TupleGetItem(targets, i);
    auto input_i = ib->TupleGetItem(inputs, i);
    nodes.push_back(ib->Reshape(input_i, ib->Shape(target_i)));
  }
  auto out = ib->MakeTuple(nodes);
  return out;
}

inline NodePtr ForEachOutZeros(BpropBuilder *ib, const NodePtr &node) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (is_dyn_seq) {
    MS_LOG(EXCEPTION) << "The tensor num of tuple[tensor] should not be dynamic, which is not supported.";
  }
  auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  auto num = sequence_shape->size();
  std::vector<NodePtr> new_nodes;
  for (size_t i = 0; i < num; ++i) {
    auto tensor_i = ib->TupleGetItem(node, i);
    new_nodes.push_back(ib->OutZeros(tensor_i));
  }
  auto zeros_node = ib->MakeTuple(new_nodes);
  return zeros_node;
}

inline std::vector<NodePtr> GMMSplitTuple(BpropBuilder *ib, const NodePtr &node, const std::vector<size_t> &nums) {
  NodePtrList split_nodes;
  size_t cur_begin = 0;
  for (size_t i = 0; i < nums.size(); i++) {
    auto cur_end = cur_begin + nums[i];
    std::vector<NodePtr> cur_nodes;
    for (size_t idx = cur_begin; idx < cur_end; idx++) {
      cur_nodes.push_back(ib->TupleGetItem(node, idx));
    }
    split_nodes.push_back(ib->MakeTuple(cur_nodes));
    // step
    cur_begin = cur_end;
  }
  return split_nodes;
}

inline std::pair<size_t, size_t> GMMBackwardParamsCheck(const std::string &op_name, const NodePtr &x,
                                                        const NodePtr &weight, const NodePtr &split_item,
                                                        const NodePtr &group_type,
                                                        const std::vector<NodePtr> &check_params,
                                                        const std::string &none_params) {
  auto num_x = FetchNumOfSequenceTensor(x, op_name, "x");
  auto num_w = FetchNumOfSequenceTensor(weight, op_name, "weight");

  CheckValueOfInt64Scalar(split_item, op_name, "split_item", SizeToLong(i3));
  CheckValueOfInt64Scalar(group_type, op_name, "group_type", SizeToLong(i0));

  auto is_none_func = [](const NodePtr &node) {
    auto type = node->abstract()->BuildType();
    return type->isa<TypeNone>();
  };
  if (!std::all_of(check_params.begin(), check_params.end(), is_none_func)) {
    MS_LOG(EXCEPTION) << "For " << op_name << "'s backward, " << none_params << " should all be None.";
  }

  return std::make_pair(num_x, num_w);
}

inline NodePtr GMMBiasBackward(BpropBuilder *ib, const NodePtr &bias, const std::string &op_name) {
  auto bias_abs = bias->abstract();
  MS_EXCEPTION_IF_NULL(bias_abs);
  auto bias_type = bias_abs->BuildType();
  auto is_bias_none = bias_type->isa<TypeNone>();
  if (bias->need_compute_grad_out() && !is_bias_none) {
    MS_LOG(EXCEPTION) << "For " << op_name << "'s backward, bias was expected to be None, but got "
                      << bias_abs->ToString();
  }
  NodePtr dbias{nullptr};
  if (is_bias_none) {
    dbias = ib->OutZeros(bias);
  } else {
    dbias = ForEachOutZeros(ib, bias);
  }
  return dbias;
}

bool CloneInplaceInputFuncForInplaceMul(const PynativeCallback &cb) {
  if (!cb.IsNotRequiresGrad(i1)) {
    return true;
  }
  return false;
}

void FreeTensorsOfInplaceMul(const PynativeCallback &cb) {
  cb.FreeOutputDeviceAddress();
  // For operators like Mul, the dx ONLY rely on y, and dy ONLY rely on x.
  // so if y is a valuenode, the dy is useless, we can free x in ahead.
  auto &inputs = *cb.GetInputs();
  if (cb.IsNotRequiresGrad(i0) && inputs[i1]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i1]);
    MS_LOG(DEBUG) << "Clear device address for inputs[1] of " << cb.opname();
  }
  if (cb.IsNotRequiresGrad(i1) && inputs[i0]->isa<tensor::Tensor>()) {
    cb.FreeDeviceAddress(&inputs[i0]);
    MS_LOG(DEBUG) << "Clear device address for inputs[0] of " << cb.opname();
  }
}

REG_BPROP_BUILDERS_BEGIN(GradMathOps)
REG_BPROP_BUILDER("MatMul").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &trans_a = ib->GetInput(i2);
  const auto &trans_b = ib->GetInput(i3);
  auto ta_opt = mindspore::GetScalarValue<bool>(trans_a->BuildValue());
  auto tb_opt = mindspore::GetScalarValue<bool>(trans_b->BuildValue());

  if (!ta_opt.has_value() || !tb_opt.has_value()) {
    MS_LOG_EXCEPTION << "For MatMul, got invalid 'transpose_a' or 'transpose_b'.";
  }
  auto ta = ta_opt.value();
  auto tb = tb_opt.value();

  auto x_type = ib->GetDtype(x);
  auto w_type = ib->GetDtype(w);
  const auto &dout = ib->GetInput(i5);
  NodePtr dx;
  NodePtr dw;

  if (((*x_type) == (*kComplex64) && (*w_type) == (*kComplex64)) ||
      ((*x_type) == (*kComplex128) && (*w_type) == (*kComplex128))) {
    // complex need conjoint transform
    if (x->need_compute_grad_out()) {
      if (ta) {
        dx = ib->MatMul(w, (ib->Emit("Conj", {dout})), (ta && tb), (ta || (!tb)));
        dx = ib->Emit("Conj", {dx});
      } else {
        dx = ib->MatMul((ib->Emit("Conj", {dout})), w, (ta && tb), (ta || (!tb)));
        dx = ib->Emit("Conj", {dx});
      }
    } else {
      dx = ib->OutZeros(x);
    }
    if (w->need_compute_grad_out()) {
      if (tb) {
        dw = ib->MatMul((ib->Emit("Conj", {dout})), x, ((!ta) || tb), (ta && tb));
        dw = ib->Emit("Conj", {dw});
      } else {
        dw = ib->MatMul((ib->Emit("Conj", {x})), dout, ((!ta) || tb), (ta && tb));
      }
    } else {
      dw = ib->OutZeros(w);
    }
    return {dx, dw, ib->OutZeros(trans_a), ib->OutZeros(trans_b)};
  }

  if ((*x_type) == (*kComplex64) || (*x_type) == (*kComplex128) || (*w_type) == (*kComplex64) ||
      (*w_type) == (*kComplex128)) {
    // only support complex64 * complex64 and complex128 * complex128, others throw exception
    MS_EXCEPTION(TypeError) << "For 'MatMul', gradient not support x_type " << x_type << " * w_type " << w_type;
  }
  if (x->need_compute_grad_out()) {
    if (ta) {
      dx = ib->MatMul(w, dout, (ta && tb), (ta || (!tb)));
    } else {
      dx = ib->MatMul(dout, w, (ta && tb), (ta || (!tb)));
    }
  } else {
    dx = ib->OutZeros(x);
  }
  if (w->need_compute_grad_out()) {
    if (tb) {
      dw = ib->MatMul(dout, x, ((!ta) || tb), (ta && tb));
    } else {
      dw = ib->MatMul(x, dout, ((!ta) || tb), (ta && tb));
    }
  } else {
    dw = ib->OutZeros(w);
  }
  return {dx, dw, ib->OutZeros(trans_a), ib->OutZeros(trans_b)};
});

ShapeArray MatMulExtBPropShapeCalcFunc(const ShapeArray &inputs) {
  auto &input_shape = inputs.at(i0);
  auto &weight_shape = inputs.at(i1);
  auto &dout_shape = inputs.at(i2);
  bool is_x_empty = IsShapeNone(input_shape);
  bool is_w_empty = IsShapeNone(weight_shape);
  if (is_x_empty || is_w_empty) {
    MS_LOG(EXCEPTION)
      << "For MatMulExt's backward, inputs must not be empty tensor for dynamic shape case in JIT mode.";
  }

  auto x_rank = input_shape.size();
  auto w_rank = weight_shape.size();
  auto dout_rank = dout_shape.size();
  ShapeVector expanded_input_shape = input_shape;
  ShapeVector expanded_weight_shape = weight_shape;
  ShapeVector expanded_dout_shape = dout_shape;
  // Transpose perm for dout before its reshaping
  ShapeVector dout_perm(dout_rank);
  std::iota(dout_perm.begin(), dout_perm.end(), 0);
  // Do prefix squeeze input and weight first
  if (x_rank > 2) {
    auto it = std::find_if(expanded_input_shape.begin(), expanded_input_shape.end(), [](int x) { return x != 1; });
    expanded_input_shape.erase(expanded_input_shape.begin(), it);
    x_rank = expanded_input_shape.size();
  }
  if (w_rank > 2) {
    auto it = std::find_if(expanded_weight_shape.begin(), expanded_weight_shape.end(), [](int x) { return x != 1; });
    expanded_weight_shape.erase(expanded_weight_shape.begin(), it);
    w_rank = expanded_weight_shape.size();
  }
  // expand input and weight shape
  if (w_rank == 1) {
    expanded_weight_shape.push_back(1);
    w_rank++;
    expanded_dout_shape.push_back(1);
    dout_rank++;
  }
  if (x_rank == 1) {
    expanded_input_shape.insert(expanded_input_shape.begin(), 1);
    x_rank++;
    expanded_dout_shape.insert(expanded_dout_shape.end() - 1, 1);
    dout_rank++;
  }
  ShapeVector x_optim_shape = expanded_input_shape;
  ShapeVector w_optim_shape = expanded_weight_shape;
  ShapeVector dout_optim_shape_for_dx = expanded_dout_shape;
  ShapeVector dout_optim_shape_for_dw = expanded_dout_shape;
  // perform transpose to w
  std::swap(w_optim_shape[w_rank - 2], w_optim_shape[w_rank - 1]);
  if (x_rank == 2 && w_rank > 2) {
    auto w_outer_dim = std::accumulate(w_optim_shape.begin(), w_optim_shape.end() - 1, static_cast<int64_t>(1),
                                       std::multiplies<int64_t>());
    w_optim_shape = {w_outer_dim, w_optim_shape[w_rank - 1]};
    auto dout_outer_dim =
      expanded_dout_shape[dout_rank - 1] * std::accumulate(expanded_dout_shape.begin(), expanded_dout_shape.end() - 2,
                                                           static_cast<int64_t>(1), std::multiplies<int64_t>());
    dout_optim_shape_for_dx = {expanded_dout_shape[dout_rank - 2], dout_outer_dim};
    auto key_dim = dout_perm[dout_shape.size() - 2];
    dout_perm.erase(dout_perm.end() - 2);
    dout_perm.insert(dout_perm.begin(), key_dim);
  }
  if (w_rank == 2 && x_rank > 2) {
    auto x_outer_dim = std::accumulate(x_optim_shape.begin(), x_optim_shape.end() - 1, static_cast<int64_t>(1),
                                       std::multiplies<int64_t>());
    x_optim_shape = {x_outer_dim, x_optim_shape[x_rank - 1]};
    auto dout_outer_dim = std::accumulate(expanded_dout_shape.begin(), expanded_dout_shape.end() - 1,
                                          static_cast<int64_t>(1), std::multiplies<int64_t>());
    dout_optim_shape_for_dw = {dout_outer_dim, expanded_dout_shape[dout_rank - 1]};
  }
  return {expanded_weight_shape,   x_optim_shape,           w_optim_shape,
          dout_optim_shape_for_dx, dout_optim_shape_for_dw, dout_perm};
}

std::vector<int64_t> MatMulExtBPropShapeCalcInferFunc(const ShapeArray &inputs, const HashSet<size_t> &) {
  int64_t expanded_input_rank = abstract::TensorShape::kShapeDimAny;
  int64_t expanded_weight_rank = abstract::TensorShape::kShapeDimAny;
  int64_t expanded_dout_rank = abstract::TensorShape::kShapeDimAny;
  int64_t x_optim_rank = abstract::TensorShape::kShapeDimAny;
  int64_t w_optim_rank = abstract::TensorShape::kShapeDimAny;
  int64_t dout_optim_rank_for_dx = abstract::TensorShape::kShapeDimAny;
  int64_t dout_optim_rank_for_dw = abstract::TensorShape::kShapeDimAny;
  int64_t dout_perm_rank = abstract::TensorShape::kShapeDimAny;

  // To ensure whether prefix squeeze can be performed.
  auto can_squeeze = [](const ShapeVector &shape) -> bool {
    for (const auto &dim : shape) {
      if (dim == abstract::TensorShape::kShapeDimAny) return false;
      if (dim != 1) return true;
    }
    return true;
  };

  if (!IsDynamicRank(inputs[0]) && !IsDynamicRank(inputs[1]) && can_squeeze(inputs[0]) && can_squeeze(inputs[1])) {
    auto input_shape = inputs.at(i0);
    auto weight_shape = inputs.at(i1);
    auto dout_shape = inputs.at(i2);
    expanded_input_rank = SizeToLong(input_shape.size());
    expanded_weight_rank = SizeToLong(weight_shape.size());
    expanded_dout_rank = SizeToLong(dout_shape.size());
    if (expanded_input_rank > 2) {
      auto it = std::find_if(input_shape.begin(), input_shape.end(), [](int x) { return x != 1; });
      input_shape.erase(input_shape.begin(), it);
      expanded_input_rank = SizeToLong(input_shape.size());
    }
    if (expanded_weight_rank > 2) {
      auto it = std::find_if(weight_shape.begin(), weight_shape.end(), [](int x) { return x != 1; });
      weight_shape.erase(weight_shape.begin(), it);
      expanded_weight_rank = SizeToLong(weight_shape.size());
    }
    if (expanded_weight_rank == 1) {
      expanded_weight_rank++;
      expanded_dout_rank++;
    }
    if (expanded_input_rank == 1) {
      expanded_input_rank++;
      expanded_dout_rank++;
    }
    x_optim_rank = expanded_input_rank;
    w_optim_rank = expanded_weight_rank;
    dout_optim_rank_for_dx = expanded_dout_rank;
    dout_optim_rank_for_dw = expanded_dout_rank;
    if (expanded_input_rank == 2 && expanded_weight_rank > 2) {
      w_optim_rank = 2;
      dout_optim_rank_for_dx = 2;
    }
    if (expanded_weight_rank == 2 && expanded_input_rank > 2) {
      x_optim_rank = 2;
      dout_optim_rank_for_dw = 2;
    }
  }
  return {expanded_weight_rank,   x_optim_rank,           w_optim_rank,
          dout_optim_rank_for_dx, dout_optim_rank_for_dw, dout_perm_rank};
}

DEF_PURE_SHAPE_CALC(g_matmul_ext_bprop_shapecalc)
  .SetCalc(MatMulExtBPropShapeCalcFunc)
  .SetInfer(MatMulExtBPropShapeCalcInferFunc);

DEF_PURE_SHAPE_CALC(g_matmul_ext_bprop_bc_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &input_shape = inputs.at(i0);
    auto &weight_shape = inputs.at(i1);
    auto &dx_shape = inputs.at(i2);
    auto &dw_shape = inputs.at(i3);
    auto dx_rank = dx_shape.size();
    auto dw_rank = dw_shape.size();
    ShapeVector dx_reduce_idx{};
    ShapeVector dw_reduce_idy{};
    if (dx_rank <= 2 || dw_rank <= 2) {
      return {dx_reduce_idx, dw_reduce_idy};
    }
    size_t ignore_offset = 2;
    return BroadcastGradientArgsInferValue(input_shape, weight_shape, ignore_offset);
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto &dx_shape = inputs.at(i2);
    auto &dw_shape = inputs.at(i3);
    auto dx_rank = dx_shape.size();
    auto dw_rank = dw_shape.size();
    if ((!IsDynamicRank(dx_shape) && dx_rank <= 2) || (!IsDynamicRank(dw_shape) && dw_rank <= 2)) {
      return {0, 0};
    }
    return {abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
  });

DEF_PURE_SHAPE_CALC(g_matmul_ext_transpose_perm_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &input_shape = inputs.at(i0);
    auto rank = input_shape.size();
    ShapeVector perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    if (rank < 2) {
      return {perm};
    }
    std::swap(perm[rank - 2], perm[rank - 1]);
    return {perm};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    return {abstract::TensorShape::kShapeDimAny};
  });

inline NodePtr MatMulInputBackwardDyn(BpropBuilder *ib, NodePtr w, NodePtr dout, const NodePtrList &shapes,
                                      const bool &is_complex) {
  auto w_expanded_shape = shapes[i0];
  auto w_optim_shape = shapes[i2];
  auto dout_optim_shape_for_dx = shapes[i3];
  auto dout_perm = shapes[i5];

  w = ib->Reshape(w, w_expanded_shape);
  // TransposeExtView is not complete for now
  auto perm = ib->ShapeCalc(g_matmul_ext_transpose_perm_shapecalc, {w});
  w = ib->Transpose(w, perm[0]);
  if (is_complex) {
    dout = ib->Emit("Conj", {dout});
  }

  w = ib->Reshape(w, w_optim_shape);
  dout = ib->Transpose(dout, dout_perm);
  dout = ib->Reshape(dout, dout_optim_shape_for_dx);
  auto dx = ib->MatMulExt(dout, w);
  return is_complex ? ib->Emit("Conj", {dx}) : dx;
}

inline NodePtr MatMulWeightBackwardDyn(BpropBuilder *ib, NodePtr x, NodePtr dout, const NodePtrList &shapes,
                                       const bool &is_complex) {
  auto x_optim_shape = shapes[i1];
  auto dout_optim_shape_for_dw = shapes[i4];
  if (is_complex) {
    x = ib->Emit("Conj", {x});
  }

  x = ib->Reshape(x, x_optim_shape);
  dout = ib->Reshape(dout, dout_optim_shape_for_dw);
  // TransposeExtView is not complete for now
  auto perm = ib->ShapeCalc(g_matmul_ext_transpose_perm_shapecalc, {x});
  x = ib->Transpose(x, perm[0]);
  auto dw = ib->MatMulExt(x, dout);
  return dw;
}

inline NodePtrList MatMulBackwardDynamic(BpropBuilder *ib, const bool &is_complex) {
  auto x = ib->GetInput(i0);
  auto w = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  NodePtrList shapes{};
  // Calculate optimized shapes used in back propagation of input or weight
  if (x->need_compute_grad_out() || w->need_compute_grad_out()) {
    shapes = ib->ShapeCalc(g_matmul_ext_bprop_shapecalc, {x, w, dout});
  }
  auto dx = x->need_compute_grad_out() ? MatMulInputBackwardDyn(ib, w, dout, shapes, is_complex) : ib->ZerosLike(x);
  auto dw = w->need_compute_grad_out() ? MatMulWeightBackwardDyn(ib, x, dout, shapes, is_complex) : ib->ZerosLike(w);
  // If the dimension of x or w larger than 2, inverse broadcasting must be took into consideration.
  const auto &dx_shape = dx->shape();
  const auto &dw_shape = dw->shape();
  if ((!IsDynamicRank(dx_shape) && dx_shape.size() <= 2) || (!IsDynamicRank(dw_shape) && dw_shape.size() <= 2)) {
    dx = ib->Reshape(dx, ib->Shape(x));
    dw = ib->Reshape(dw, ib->Shape(w));
    return {dx, dw};
  }
  // Calculate the reduce axis of inverse broadcast
  auto bc_axis = ib->ShapeCalc(g_matmul_ext_bprop_bc_shapecalc, {x, w, dx, dw});
  // SumExt is not adopted, since the 'skip mode' in ReduceSum is necessary for dynamic rank case.
  dx = ib->ReduceSum(dx, bc_axis[0], false, true);
  dx = ib->Reshape(dx, ib->Shape(x));
  dw = ib->ReduceSum(dw, bc_axis[1], false, true);
  dw = ib->Reshape(dw, ib->Shape(w));
  return {dx, dw};
}

inline NodePtr MatMulInputBackward(BpropBuilder *ib, const bool &is_complex) {
  auto x = ib->GetInput(i0);
  auto w = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  const auto &ori_x_shape = x->shape();
  auto ori_x_rank = ori_x_shape.size();
  // squeeze x
  if (ori_x_rank > 2 && ori_x_shape[0] == 1) {
    ShapeVector sqz_shape = ori_x_shape;
    auto it = std::find_if(sqz_shape.begin(), sqz_shape.end(), [](int x) { return x != 1; });
    sqz_shape.erase(sqz_shape.begin(), it);
    x = ib->Reshape(x, sqz_shape);
  }

  if (w->shape().size() == 1) {
    w = ib->ExpandDims(w, -1);
    dout = ib->ExpandDims(dout, -1);
  }
  if (x->shape().size() == 1) {
    x = ib->ExpandDims(x, 0);
    dout = ib->ExpandDims(dout, -2);
  }

  // TransposeExtView is not complete for now
  w = MatrixTransposeExt(ib, w);
  if (is_complex) {
    dout = ib->Emit("Conj", {dout});
  }

  const auto &x_shape = x->shape();
  const auto &w_shape = w->shape();
  const auto &dout_shape = dout->shape();
  auto x_rank = x_shape.size();
  auto w_rank = w_shape.size();
  auto dout_rank = dout_shape.size();
  NodePtr dx = nullptr;
  if (x_rank == 2 && w_rank > 2) {
    w = ib->Reshape(w, {-1, w_shape[w_rank - 1]});
    std::vector<int64_t> perm(dout_rank);
    std::iota(perm.begin(), perm.end(), 0);
    auto key_dim = perm[dout_rank - 2];
    perm.erase(perm.end() - 2);
    perm.insert(perm.begin(), key_dim);
    dout = ib->Transpose(dout, perm);
    dout = ib->Reshape(dout, {dout_shape[dout_rank - 2], -1});
    dx = ib->MatMulExt(dout, w);
    dx = ib->Reshape(dx, ori_x_shape);
  } else {
    dx = ib->MatMulExt(dout, w);
    const auto &dx_shape = dx->shape();
    auto dx_rank = dx_shape.size();
    if (ori_x_rank == 1 && dx_rank != 1 && dx_shape[0] == 1) {
      dx = ib->Reshape(dx, ori_x_shape);
    }
  }
  return is_complex ? ib->Emit("Conj", {dx}) : dx;
}

inline NodePtr MatMulWeightBackward(BpropBuilder *ib, const bool &is_complex) {
  auto x = ib->GetInput(i0);
  auto w = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  const auto &ori_w_shape = w->shape();
  auto ori_w_rank = ori_w_shape.size();
  // squeeze w
  if (ori_w_rank > 2 && ori_w_shape[0] == 1) {
    ShapeVector sqz_shape = ori_w_shape;
    auto it = std::find_if(sqz_shape.begin(), sqz_shape.end(), [](int x) { return x != 1; });
    sqz_shape.erase(sqz_shape.begin(), it);
    w = ib->Reshape(w, sqz_shape);
  }

  if (w->shape().size() == 1) {
    w = ib->ExpandDims(w, -1);
    dout = ib->ExpandDims(dout, -1);
  }
  if (x->shape().size() == 1) {
    x = ib->ExpandDims(x, 0);
    dout = ib->ExpandDims(dout, -2);
  }

  if (is_complex) {
    x = ib->Emit("Conj", {x});
  }

  const auto &x_shape = x->shape();
  const auto &w_shape = w->shape();
  const auto &dout_shape = dout->shape();
  auto x_rank = x_shape.size();
  auto w_rank = w_shape.size();
  auto dout_rank = dout_shape.size();
  NodePtr dw = nullptr;
  if (w_rank == 2 && x_rank > 2) {
    x = ib->Reshape(x, {-1, x_shape[x_rank - 1]});
    dout = ib->Reshape(dout, {-1, dout_shape[dout_rank - 1]});
    x = MatrixTransposeExt(ib, x);
    dw = ib->MatMulExt(x, dout);
    dw = ib->Reshape(dw, ori_w_shape);
  } else {
    x = MatrixTransposeExt(ib, x);
    dw = ib->MatMulExt(x, dout);
    const auto &dw_shape = dw->shape();
    auto dw_rank = dw_shape.size();
    if (ori_w_rank == 1 && dw_rank != 1 && dw_shape[dw_rank - 1] == 1) {
      dw = ib->Reshape(dw, ori_w_shape);
    }
  }
  return dw;
}

REG_BPROP_BUILDER("MatMulExt").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &x_shape = x->shape();
  const auto &w_shape = w->shape();
  NodePtr dx = nullptr;
  NodePtr dw = nullptr;
  bool is_x_empty = IsShapeNone(x_shape);
  bool is_w_empty = IsShapeNone(w_shape);
  if (is_x_empty || is_w_empty) {
    dx = x->need_compute_grad_out() ? (is_x_empty ? x : ib->OutZeros(x)) : ib->OutZeros(x);
    dw = w->need_compute_grad_out() ? (is_w_empty ? w : ib->OutZeros(w)) : ib->OutZeros(x);
    return {dx, dw};
  }
  auto x_type = ib->GetDtype(x);
  auto w_type = ib->GetDtype(w);
  bool is_complex = false;
  if ((*x_type) == (*kComplex64) || (*x_type) == (*kComplex128) || (*w_type) == (*kComplex64) ||
      (*w_type) == (*kComplex128)) {
    if ((*x_type) != (*w_type)) {
      // only support complex64 * complex64 and complex128 * complex128, others throw exception
      MS_EXCEPTION(TypeError)
        << "For 'MatMulExt', in complex calculation case, the data type of [input] and [other] must"
           " be both complex64 or complex128, but got: "
        << x_type << " vs " << w_type;
    } else {
      is_complex = true;
    }
  }
  // handle dynamic case
  NodePtrList ret{};
  bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
  bool is_dynamic_shape = IsDynamicShape(x_shape) || IsDynamicShape(w_shape);
  if (is_dynamic_rank || is_dynamic_shape) {
    return MatMulBackwardDynamic(ib, is_complex);
  }
  dx = x->need_compute_grad_out() ? MatMulInputBackward(ib, is_complex) : ib->OutZeros(x);
  dw = w->need_compute_grad_out() ? MatMulWeightBackward(ib, is_complex) : ib->OutZeros(w);
  const auto &dx_shape = dx->shape();
  const auto &dw_shape = dw->shape();
  if (dx_shape.size() <= 2 || dw_shape.size() <= 2) {
    return {dx, dw};
  }
  if (!x->need_compute_grad_out()) {
    dx = nullptr;
  }
  if (!w->need_compute_grad_out()) {
    dw = nullptr;
  }
  return BinopGradCommon(ib, x, w, dx, dw, 2);
});

REG_BPROP_BUILDER("Mm").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &mat2 = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx;
  NodePtr dw;
  if (input->need_compute_grad_out()) {
    auto mat2_t = MatrixTransposeExt(ib, mat2);
    dx = ib->Mm(dout, mat2_t);
  } else {
    dx = ib->OutZeros(input);
  }
  if (mat2->need_compute_grad_out()) {
    auto input_t = MatrixTransposeExt(ib, input);
    dw = ib->Mm(input_t, dout);
  } else {
    dw = ib->OutZeros(mat2);
  }
  return {dx, dw};
});

REG_BPROP_BUILDER("GroupedMatmul").SetUnusedInputs({i2, i3, i4, i5, i6, i12}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &weight = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &group_list = ib->GetInput(i7);
  const auto &split_item = ib->GetInput(i8);
  const auto &group_type = ib->GetInput(i9);
  const auto &transpose_a = ib->GetInput(i10);
  const auto &transpose_b = ib->GetInput(i11);
  const auto &dout = ib->GetInput(i13);
  // none_params
  const auto &scale = ib->GetInput(i3);
  const auto &offset = ib->GetInput(i4);
  const auto &antiquant_scale = ib->GetInput(i5);
  const auto &antiquant_offset = ib->GetInput(i6);

  const std::string op_name = "GroupedMatmul";
  auto transpose_a_opt = GetScalarValue<bool>(transpose_a->BuildValue());
  auto transpose_b_opt = GetScalarValue<bool>(transpose_b->BuildValue());
  if ((!transpose_a_opt.has_value() || transpose_a_opt.value()) ||
      (!transpose_b_opt.has_value() || transpose_b_opt.value())) {
    MS_EXCEPTION(ValueError) << "For " << op_name << "'s, backward, transpose_a and transpose_b should both be false.";
  }
  const std::string none_params = "scale, offset, antiquant_scale and antiquant_offset";
  auto [num_x, num_w] = GMMBackwardParamsCheck(op_name, x, weight, split_item, group_type,
                                               {scale, offset, antiquant_scale, antiquant_offset}, none_params);

  auto gmm_func = [&ib, &op_name, &transpose_a, &transpose_b](const NodePtr &x, const NodePtr &weight,
                                                              const NodePtr &group_list, int64_t group_type) {
    auto none_node = ib->EmitValue(mindspore::kNone);
    auto split_item = ib->Value<int64_t>(3);
    return ib->Emit(op_name, {x, weight, none_node, none_node, none_node, none_node, none_node, group_list, split_item,
                              ib->Value<int64_t>(group_type), transpose_a, transpose_b});
  };

  NodePtr dx{nullptr};
  if (x->need_compute_grad_out()) {
    auto wt = ForEachTransposeLastTwoDim(ib, weight, num_w);
    dx = gmm_func(dout, wt, group_list, 0);
  } else {
    dx = ForEachOutZeros(ib, x);
  }

  NodePtr dw{nullptr};
  if (weight->need_compute_grad_out()) {
    auto xt = ForEachTransposeLastTwoDim(ib, x, num_x);
    auto dw_tmp = gmm_func(xt, dout, group_list, 2);
    dw = ForEachReshapeAs(ib, dw_tmp, weight, num_w);
  } else {
    dw = ForEachOutZeros(ib, weight);
  }

  auto dbias = GMMBiasBackward(ib, bias, op_name);

  return {dx,
          dw,
          dbias,
          ib->OutZeros(scale),
          ib->OutZeros(offset),
          ib->OutZeros(antiquant_scale),
          ib->OutZeros(antiquant_offset),
          ib->OutZeros(group_list),
          ib->OutZeros(split_item),
          ib->OutZeros(group_type),
          ib->OutZeros(transpose_a),
          ib->OutZeros(transpose_b)};
});

REG_BPROP_BUILDER("GroupedMatmulV2").SetUnusedInputs({i2, i3, i4, i5, i6, i10}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &weight = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &group_list = ib->GetInput(i7);
  const auto &split_item = ib->GetInput(i8);
  const auto &group_type = ib->GetInput(i9);
  const auto &dout = ib->GetInput(i11);
  // none params
  const auto &scale = ib->GetInput(i3);
  const auto &offset = ib->GetInput(i4);
  const auto &antiquant_scale = ib->GetInput(i5);
  const auto &antiquant_offset = ib->GetInput(i6);

  const std::string op_name = "GroupedMatmulV2";
  const std::string none_params = "scale, offset, antiquant_scale and antiquant_offset";
  auto [num_x, num_w] = GMMBackwardParamsCheck(op_name, x, weight, split_item, group_type,
                                               {scale, offset, antiquant_scale, antiquant_offset}, none_params);

  auto gradients = ib->GmmBackward(dout, x, weight, group_list, ib->Value<int64_t>(0));
  auto split_nodes = GMMSplitTuple(ib, gradients, {num_x, num_w});
  auto dx = split_nodes[i0];
  auto dw = split_nodes[i1];
  auto dbias = GMMBiasBackward(ib, bias, op_name);

  return {dx,
          dw,
          dbias,
          ib->OutZeros(scale),
          ib->OutZeros(offset),
          ib->OutZeros(antiquant_scale),
          ib->OutZeros(antiquant_offset),
          ib->OutZeros(group_list),
          ib->OutZeros(split_item),
          ib->OutZeros(group_type)};
});

REG_BPROP_BUILDER("GroupedMatmulV4")
  .SetUnusedInputs({i2, i3, i4, i5, i6, i7, i9, i10, i11, i16, i17})
  .SetBody(BODYFUNC(ib) {
    auto x = ib->GetInput(i0);
    auto weight = ib->GetInput(i1);
    auto bias = ib->GetInput(i2);
    auto group_list = ib->GetInput(i8);
    auto split_item = ib->GetInput(i12);
    auto group_type = ib->GetInput(i13);
    auto group_list_type = ib->GetInput(i14);
    auto act_type = ib->GetInput(i15);
    auto dout = ib->GetInput(i18);
    // none params
    std::vector<NodePtr> check_params;
    for (size_t i = i3; i < i12; i++) {
      if (i == i8) {
        continue;
      }
      check_params.push_back(ib->GetInput(i));
    }

    const std::string none_params =
      "scale, offset, antiquant_scale, antiquant_offset, pre_token_scale, activation_input, activation_quant_scale and "
      "activation_quant_offset";
    const std::string op_name = "GroupedMatmulV4";
    auto [num_x, num_w] = GMMBackwardParamsCheck(op_name, x, weight, split_item, group_type, check_params, none_params);

    auto dx_and_dw = ib->GmmV2Backward(dout, x, weight, group_list, group_list_type);
    auto split_nodes = GMMSplitTuple(ib, dx_and_dw, {num_x, num_w});
    auto dx = split_nodes[i0];
    auto dw = split_nodes[i1];
    auto dbias = GMMBiasBackward(ib, bias, op_name);

    const auto &group_list_shape = group_list->shape();
    NodePtr dgroup_list;
    if (!IsDynamic(group_list_shape)) {
      dgroup_list = ib->OutZeros(group_list);
    } else {
      dgroup_list = ib->Zeros(ib->Shape(group_list), ib->Value(static_cast<int64_t>(ib->GetDtypeId(group_list))));
    }

    std::vector<NodePtr> gradients{dx, dw, dbias};
    const auto &inputs = ib->GetInputs();
    std::transform(inputs.begin() + i3, inputs.begin() + i8, std::back_inserter(gradients),
                   [&ib](const NodePtr &node) { return ib->OutZeros(node); });
    gradients.push_back(std::move(dgroup_list));
    std::transform(inputs.begin() + i9, inputs.begin() + i17, std::back_inserter(gradients),
                   [&ib](const NodePtr &node) { return ib->OutZeros(node); });

    return gradients;
  });

REG_BPROP_BUILDER("Add").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = dout;
  }
  ib->MarkSharedGradTensor(dx, dy);
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("AddExt").FreeUselessValues_IO({i0, i1}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);

  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;

  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    auto alpha_opt = GetAlpha(alpha);
    if (!alpha_opt.has_value()) {
      dy = ib->Muls(dout, alpha);
    } else {
      if (alpha_opt.value() == 1) {
        dy = dout;
      } else {
        dy = ib->Muls(dout, alpha);
      }
    }
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dx, dy);
  auto dx_cast = x->need_compute_grad_out() ? ib->Cast(ret[0], ib->GetDtype(x)) : ib->OutZeros(x);
  auto dy_cast = y->need_compute_grad_out() ? ib->Cast(ret[1], ib->GetDtype(y)) : ib->OutZeros(y);
  ib->MarkSharedGradTensor(dx, dy);
  return {dx_cast, dy_cast, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("AddScalar").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  auto dtype_input = ib->GetDtype(input);
  bool is_complex_input = (*dtype_input) == (*kComplex64) || (*dtype_input) == (*kComplex128);

  auto dtype_dout = ib->GetDtype(dout);
  bool is_complex_dout = (*dtype_dout) == (*kComplex64) || (*dtype_dout) == (*kComplex128);

  if (!is_complex_input && is_complex_dout) {
    return {ib->Real(dout), ib->OutZeros(other), ib->OutZeros(alpha)};
  }

  return {dout, ib->OutZeros(other), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceAddsExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;

  if (x->need_compute_grad_out()) {
    dx = dout;
  } else {
    dx = ib->OutZeros(x);
  }

  return {dx, ib->OutZeros(y), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceAddExt").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr dy = nullptr;
  NodePtr dx = nullptr;

  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = dout;
    auto alpha_opt = GetAlpha(alpha);
    if (!alpha_opt.has_value()) {
      auto alpha_tensor = ib->ScalarToTensor(alpha, ib->GetDtype(x));
      dy = ib->Mul(dy, alpha_tensor);
    } else if (alpha_opt.value() != 1) {
      auto alpha_tensor = ib->ValueByType(alpha_opt.value(), ib->GetDtype(x));
      dy = ib->Muls(dy, alpha_tensor);
    }
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dx, dy);
  auto input_cast = x->need_compute_grad_out() ? ib->Cast(ret[0], ib->GetDtype(x)) : ib->OutZeros(x);
  auto other_cast = y->need_compute_grad_out() ? ib->Cast(ret[1], ib->GetDtype(y)) : ib->OutZeros(y);
  ib->MarkSharedGradTensor(input_cast, other_cast);
  return {input_cast, other_cast, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceMul")
  .FreeUselessValues(FreeTensorsOfInplaceMul)
  .CloneInplaceInput(CloneInplaceInputFuncForInplaceMul)
  .SetBody(BODYFUNC(ib) {
    auto x = ib->GetInput(i0);
    auto y = ib->GetInput(i1);
    auto dout = ib->GetInput(i3);
    NodePtr bc_dx = nullptr;
    NodePtr bc_dy = nullptr;
    if (x->need_compute_grad_out()) {
      bc_dx = ib->Mul(y, dout);
    }
    if (y->need_compute_grad_out()) {
      bc_dy = ib->Mul(x, dout);
    }
    auto ret = BinopGradCommon(ib, x, y, bc_dx, bc_dy);
    auto dx = x->need_compute_grad_out() ? ib->Cast(ret[i0], ib->GetDtype(x)) : ret[i0];
    auto dy = y->need_compute_grad_out() ? ib->Cast(ret[i1], ib->GetDtype(y)) : ret[i1];
    return {dx, dy};
  });

REG_BPROP_BUILDER("InplaceMuls").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Muls(dout, y);
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("SubScalar").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  auto dtype_input = ib->GetDtype(input);
  bool is_complex_input = (*dtype_input) == (*kComplex64) || (*dtype_input) == (*kComplex128);

  auto dtype_dout = ib->GetDtype(dout);
  bool is_complex_dout = (*dtype_dout) == (*kComplex64) || (*dtype_dout) == (*kComplex128);

  if (!is_complex_input && is_complex_dout) {
    return {ib->Real(dout), ib->OutZeros(other), ib->OutZeros(alpha)};
  }

  return {dout, ib->OutZeros(other), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("SubExt").FreeUselessValues_IO({i0, i1}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);

  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;

  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    auto alpha_opt = GetAlpha(alpha);
    if (!alpha_opt.has_value()) {
      dy = ib->Muls(ib->Neg(dout), alpha);
    } else {
      if (alpha_opt.value() == 1) {
        dy = ib->Neg(dout);
      } else {
        dy = ib->Muls(ib->Neg(dout), alpha);
      }
    }
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dx, dy);
  auto dx_cast = x->need_compute_grad_out() ? ib->Cast(ret[0], ib->GetDtype(x)) : ib->OutZeros(x);
  auto dy_cast = y->need_compute_grad_out() ? ib->Cast(ret[1], ib->GetDtype(y)) : ib->OutZeros(y);
  return {dx_cast, dy_cast, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Mul").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Mul', gradient not support for complex type currently.";
  }
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Emit("Mul", {y, dout});
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Emit("Mul", {x, dout});
  }
  auto ret = BinopGradCommon(ib, x, y, bc_dx, bc_dy);
  auto dx = ret[i0];
  if (x->need_compute_grad_out()) {
    auto x_type = ib->GetDtype(x);
    if (ib->GetDtypeId(dx) != x_type->type_id()) {
      dx = ib->Cast(dx, x_type);
    }
  }
  auto dy = ret[i1];
  if (y->need_compute_grad_out()) {
    auto y_type = ib->GetDtype(y);
    if (ib->GetDtypeId(dy) != y_type->type_id()) {
      dy = ib->Cast(dy, y_type);
    }
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("Muls").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &alpha = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_dtype = ib->GetDtype(x);

  NodePtr bc_dx = nullptr;
  bc_dx = ib->Muls(dout, alpha);
  auto dx = bc_dx;
  if (ib->GetDtypeId(dx) != x_dtype->type_id()) {
    dx = ib->Cast(dx, x_dtype);
  }
  return {dx, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Sub").FreeUselessValues_IO({i0, i1}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = ib->Neg(dout);
  }
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("InplaceSubExt").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr input_bc = nullptr;
  NodePtr other_bc = nullptr;

  if (input->need_compute_grad_out()) {
    input_bc = dout;
  }

  if (other->need_compute_grad_out()) {
    auto alpha_opt = GetAlpha(alpha);
    if (!alpha_opt.has_value()) {
      other_bc = ib->Neg(dout);
      auto alpha_tensor = ib->ScalarToTensor(alpha, ib->GetDtype(input));
      other_bc = ib->Mul(other_bc, alpha_tensor);
    } else {
      auto tolerance = 1e-9;
      auto alpha_dtype = ib->GetDtypeId(alpha);
      if ((alpha_dtype == kNumberTypeInt64 && alpha_opt.value() == 1) ||
          (alpha_dtype == kNumberTypeFloat32 && fabs(alpha_opt.value() - 1) < tolerance)) {
        other_bc = ib->Neg(dout);
      } else {
        other_bc = ib->Neg(dout);
        other_bc = ib->Muls(other_bc, alpha);
      }
    }
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, input, other, input_bc, other_bc);
  auto input_cast = input->need_compute_grad_out() ? ib->Cast(ret[0], ib->GetDtype(input)) : ib->OutZeros(input);
  auto other_cast = other->need_compute_grad_out() ? ib->Cast(ret[1], ib->GetDtype(other)) : ib->OutZeros(other);
  return {input_cast, other_cast, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceSubScalar").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &alpha = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr input_bc = nullptr;

  input_bc = ib->Cast(dout, ib->GetDtype(input));
  return {input_bc, ib->OutZeros(other), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("FmodTensor").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto self_div = ib->DivMod(input, other, ops::RoundingMode::TRUNC);
  auto other_dout = ib->Neg(ib->Mul(dout, self_div));
  return BinopGradCommon(ib, input, other, dout, other_dout);
});

REG_BPROP_BUILDER("FmodScalar").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {ib->Cast(dout, ib->GetDtype(input)), ib->OutZeros(other)};
});

REG_BPROP_BUILDER("InplaceDiv")
  .FreeUselessValues(FreeTensorOfInplaceDivTensor)
  .CloneInplaceInput(CloneInplaceInputFuncForInplaceDiv)
  .SetBody(BODYFUNC(ib) {
    auto input = ib->GetInput(i0);
    auto other = ib->GetInput(i1);
    auto dout = ib->GetInput(i3);
    NodePtr bc_input = nullptr;
    NodePtr bc_other = nullptr;
    if (input->need_compute_grad_out()) {
      auto other_type = ib->GetDtypeId(other);
      if (other_type == kNumberTypeComplex64 || other_type == kNumberTypeComplex128) {
        auto other_conj = ib->Conj(other);
        bc_input = ib->Div(dout, other_conj);
      } else {
        bc_input = ib->Div(dout, other);
      }
    }

    if (other->need_compute_grad_out()) {
      auto div_res1 = ib->Div(input, other);
      auto div_res2 = ib->Div(div_res1, other);
      auto div_res2_type = ib->GetDtypeId(div_res2);
      if (div_res2_type == kNumberTypeComplex64 || div_res2_type == kNumberTypeComplex128) {
        div_res2 = ib->Conj(div_res2);
      }
      bc_other = -dout * div_res2;
    }

    std::vector<NodePtr> ret = BinopGradCommon(ib, input, other, bc_input, bc_other);
    auto input_cast = input->need_compute_grad_out() ? ib->Cast(ret[0], ib->GetDtype(input)) : ib->OutZeros(input);
    auto other_cast = other->need_compute_grad_out() ? ib->Cast(ret[1], ib->GetDtype(other)) : ib->OutZeros(other);
    return {input_cast, other_cast};
  });

REG_BPROP_BUILDER("InplaceDivs").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  NodePtr bc_input = nullptr;
  auto other_type = ib->GetDtypeId(other);
  if (other_type == kNumberTypeComplex64 || other_type == kNumberTypeComplex128) {
    other = ib->Conj(other);
  }
  bc_input = ib->Divs(dout, other);

  auto input_cast = ib->Cast(bc_input, ib->GetDtype(input));
  return {input_cast, ib->OutZeros(other)};
});

REG_BPROP_BUILDER("InplaceDivMod").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &rounding_mode = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  NodePtr bc_input = nullptr;
  NodePtr bc_other = nullptr;
  if (input->need_compute_grad_out()) {
    bc_input = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  }

  if (other->need_compute_grad_out()) {
    bc_other = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(other))));
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, input, other, bc_input, bc_other);
  ret.emplace_back(ib->OutZeros(rounding_mode));
  return ret;
});

REG_BPROP_BUILDER("InplaceDivMods").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &rounding_mode = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  NodePtr bc_input = nullptr;
  bc_input = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));

  return {bc_input, ib->OutZeros(other), ib->OutZeros(rounding_mode)};
});

REG_BPROP_BUILDER("Div").FreeUselessValues(FreeTensorsOfDiv).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  auto x_dtype_id = ib->GetDtypeId(x);
  bc_dx = ib->Div(dout, y);
  if (y->need_compute_grad_out()) {
    bc_dy = -dout * ib->Div(out, y);
  }
  auto result = BinopGradCommon(ib, x, y, bc_dx, bc_dy);
  bool is_complex = (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128);
  if (is_complex) {
    result[i0] = ib->Conj(result[i0]);
    result[i1] = y->need_compute_grad_out() ? ib->Conj(result[i1]) : ib->OutZeros(y);
  }
  return result;
});

REG_BPROP_BUILDER("Divs").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &other = ib->GetInput(i1);
  const auto &grad = ib->GetInput(i3);
  auto dinput = ib->Divs(grad, other);
  return {dinput, ib->OutZeros(other)};
});

REG_BPROP_BUILDER("DivMod").FreeUselessValues_O().SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &rounding_mode = ib->GetInput(i2);

  auto mode_type = rounding_mode->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(mode_type);

  if (mode_type->isa<TypeNone>()) {
    const auto &dout = ib->GetInput(i4);

    NodePtr bc_dx = nullptr;
    NodePtr bc_dy = nullptr;
    if (x->need_compute_grad_out()) {
      bc_dx = ib->Div(dout, y);
    }
    if (y->need_compute_grad_out()) {
      bc_dy = ib->Mul(-dout, ib->Div(ib->Div(x, y), y));
    }
    std::vector<NodePtr> result = BinopGradCommon(ib, x, y, bc_dx, bc_dy);

    auto x_dtype_id = ib->GetDtypeId(x);
    bool is_complex = (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128);
    if (is_complex) {
      result[i0] = x->need_compute_grad_out() ? ib->Conj(result[i0]) : ib->OutZeros(x);
      result[i1] = y->need_compute_grad_out() ? ib->Conj(result[i1]) : ib->OutZeros(y);
    }
    result.emplace_back(ib->OutZeros(rounding_mode));
    return result;
  } else {
    return {ib->OutZeros(x), ib->OutZeros(y), ib->OutZeros(rounding_mode)};
  }
});

REG_BPROP_BUILDER("DivMods").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &rounding_mode = ib->GetInput(i2);
  const auto &grad = ib->GetInput(i4);

  auto mode_type = rounding_mode->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(mode_type);

  NodePtr dinput{nullptr};
  if (mode_type->isa<TypeNone>()) {
    dinput = ib->Divs(grad, other);
  } else {
    dinput = ib->Zeros(ib->Shape(grad), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  }

  return {dinput, ib->OutZeros(other), ib->OutZeros(rounding_mode)};
});

REG_BPROP_BUILDER("BitwiseAnd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseAndScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseAndTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseOr").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseOrScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseOrTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseXor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseXorScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseXorTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceSub").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceAdd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceUpdate").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceUpdateV2").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Less").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LessEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalNot").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalAnd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalOr").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("AssignAdd").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dy = nullptr;
  NodePtr dx = nullptr;

  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = dout;
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dx, dy);
  auto input_cast = x->need_compute_grad_out() ? ib->Cast(ret[i0], ib->GetDtype(x)) : ib->OutZeros(x);
  auto other_cast = y->need_compute_grad_out() ? ib->Cast(ret[i1], ib->GetDtype(y)) : ib->OutZeros(y);
  return {input_cast, other_cast};
});

REG_BPROP_BUILDER("AssignSub").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = ib->Neg(dout);
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dx, dy);
  auto input_cast = x->need_compute_grad_out() ? ib->Cast(ret[i0], ib->GetDtype(x)) : ib->OutZeros(x);
  auto other_cast = y->need_compute_grad_out() ? ib->Cast(ret[i1], ib->GetDtype(y)) : ib->OutZeros(y);
  return {input_cast, other_cast};
});

REG_BPROP_BUILDER("Sin").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, ib->Conj(ib->Cos(x)));
  return {dx};
});

REG_BPROP_BUILDER("Asin").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("AsinGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinExt").FreeUselessValues_O().SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Asin', gradient not support for complex type currently.";
  } else {
    dx = dout * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  }
  return {dx};
});

REG_BPROP_BUILDER("AsinGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr d2x;
  if (x->need_compute_grad_out()) {
    auto one = ib->Tensor(1, ib->GetDtype(x));
    auto minus_one_p5 = ib->ValueByType(-1.5, ib->GetDtype(x));
    d2x =
      ib->Mul((ib->Mul((ib->Mul(dout, grad)), x)), (ib->PowTensorScalar(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  } else {
    d2x = ib->OutZeros(x);
  }
  auto ddy = grad->need_compute_grad_out() ? ib->Emit("AsinGrad", {x, dout}) : ib->OutZeros(grad);
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Asinh").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("AsinhGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinhExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Asinh', gradient not support for complex type currently.";
  } else {
    dx = dout * ib->Rsqrt(ib->AddScalar(ib->Square(x), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1)));
  }
  return {dx};
});

REG_BPROP_BUILDER("AsinhGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto minus_one = ib->Value(-1.0);
    dy = ib->Mul(ib->Muls(ib->Mul(dout, out), minus_one), ib->Tanh(y));
  } else {
    dy = ib->OutZeros(y);
  }

  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("AsinhGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Sinh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto conj_x = ib->Conj(x);
  auto dx = ib->Mul((ib->Cosh(conj_x)), dout);
  return {dx};
});

REG_BPROP_BUILDER("Cos").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, (ib->Neg(ib->Sin(x))));
  return {dx};
});

REG_BPROP_BUILDER("ACos").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("ACosGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AcosExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Acos', gradient not support for complex type currently.";
  } else {
    dx = ib->Neg(dout) * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  }
  return {dx};
});

REG_BPROP_BUILDER("ACosGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr d2x;
  if (x->need_compute_grad_out()) {
    auto one = ib->Tensor(1, ib->GetDtype(x));
    auto minus_one_p5 = ib->ValueByType(-1.5, ib->GetDtype(x));
    d2x = ib->Mul((ib->Mul((ib->Mul(ib->Neg(dout), grad)), x)),
                  (ib->PowTensorScalar(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  } else {
    d2x = ib->OutZeros(x);
  }
  auto ddy = grad->need_compute_grad_out() ? ib->Emit("ACosGrad", {x, dout}) : ib->OutZeros(grad);
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Acosh").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("AcoshGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AcoshExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Acosh', gradient not support for complex type currently.";
  } else {
    dx = dout * ib->Rsqrt(ib->SubScalar(ib->Square(x), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1)));
  }
  return {dx};
});

REG_BPROP_BUILDER("AcoshGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dy = y->need_compute_grad_out()
              ? ib->RealDiv((ib->Muls((ib->Mul(dout, out)), ib->ValueByType(-1.0, ib->GetDtype(out)))), (ib->Tanh(y)))
              : ib->OutZeros(y);
  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("AcoshGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Cosh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto conj_x = ib->Conj(x);
  auto dx = ib->Mul((ib->Sinh(conj_x)), dout);
  return {dx};
});

REG_BPROP_BUILDER("Abs").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, ib->Sign(x));
  return {dx};
});

REG_BPROP_BUILDER("Conj").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Conj(dout);
  return {dx};
});

REG_BPROP_BUILDER("ScalarCast").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &t = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  if (x->abstract()->isa<abstract::AbstractTensor>()) {
    auto dx = ib->Emit("ScalarToTensor", {dout, ib->Value<int64_t>(ib->GetDtype(x)->type_id())});
    return {dx, ib->OutZeros(t)};
  }
  auto dx = ib->Emit("ScalarCast", {dout, ib->Value<int64_t>(ib->GetDtypeId(x))});
  return {dx, ib->OutZeros(t)};
});

REG_BPROP_BUILDER("Sign").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceSign").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Round").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Atan2").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto tmp = ib->RealDiv(dout, (ib->Add((ib->Square(x)), (ib->Square(y)))));
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Mul(tmp, y);
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Mul(tmp, (ib->Neg(x)));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Atan2Ext").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  auto tmp = ib->Div(dout, ib->AddExt(ib->Square(x), ib->Square(y), ib->Value<int64_t>(1)));

  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Mul(tmp, y);
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Neg(ib->Mul(tmp, x));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("BesselI0e").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, (ib->Sub((ib->Emit("BesselI1e", {x})), (ib->Mul((ib->Sign(x)), out)))));
  return {dx};
});

REG_BPROP_BUILDER("Atan").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("AtanGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AtanExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Atan', gradient not support for complex type currently.";
  } else {
    dx = ib->Div(dout, ib->AddScalar(ib->Square(x), ib->Value(1), ib->Value(1)));
  }
  return {dx};
});

REG_BPROP_BUILDER("AtanGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dgrad = ib->Emit("AtanGrad", {x, dout});
  auto dx = x->need_compute_grad_out()
              ? ib->Mul((ib->Muls((ib->Mul(out, dgrad)), ib->ValueByType(-2.0, ib->GetDtype(x)))), x)
              : ib->OutZeros(x);
  return {dx, dgrad};
});

REG_BPROP_BUILDER("Log1p").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_1p = ib->AddScalar(x, ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  TypeId exp_type = ib->GetDtypeId(x);
  if (exp_type == kNumberTypeComplex64 || exp_type == kNumberTypeComplex128) {
    x_1p = ib->Conj(x_1p);
  }
  auto dx = ib->Div(dout, x_1p);
  return {dx};
});

REG_BPROP_BUILDER("LogAddExp").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto exp_x_y_1p = ib->AddScalar(ib->Exp(ib->Sub(x, y)), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  auto exp_y_x_1p = ib->AddScalar(ib->Exp(ib->Sub(y, x)), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  auto dx = ib->Div(dout, exp_y_x_1p);
  auto dy = ib->Div(dout, exp_x_y_1p);
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("LogAddExp2").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto exp_x_y_1p =
    ib->AddScalar(ib->PowScalarTensor(ib->Value(2), ib->Sub(x, y)), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  auto exp_y_x_1p =
    ib->AddScalar(ib->PowScalarTensor(ib->Value(2), ib->Sub(y, x)), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  auto dx = ib->Div(dout, exp_y_x_1p);
  auto dy = ib->Div(dout, exp_x_y_1p);
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("LogSumExp").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &keepdim = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);

  NodePtr dx;
  auto keepdim_opt = mindspore::GetScalarValue<bool>(keepdim->BuildValue());
  if (!keepdim_opt.has_value()) {
    auto dx_true_branch = [&](Emitter *e) -> NodePtrList { return {LogSumExpGrad(e, input, dim, true, out, dout)}; };
    auto dx_false_branch = [&](Emitter *e) -> NodePtrList { return {LogSumExpGrad(e, input, dim, false, out, dout)}; };
    auto keepdim_true = ib->Equal(keepdim, ib->Value<bool>(true));
    dx = ib->Conditional(keepdim_true, dx_true_branch, dx_false_branch);
  } else {
    dx = LogSumExpGrad(ib, input, dim, keepdim_opt.value(), out, dout);
  }
  return {dx, ib->OutZeros(dim), ib->OutZeros(keepdim)};
});

REG_BPROP_BUILDER("Erf").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto half_root_pi = ib->ValueByType(2 / sqrt(pi), ib->GetDtype(x));
  auto x_square = ib->Square(x);
  auto dx = ib->Mul((ib->Muls(dout, half_root_pi)), (ib->Exp(ib->Neg(x_square))));
  return {dx};
});

REG_BPROP_BUILDER("Erfc").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto half_root_pi = ib->ValueByType(-2 / sqrt(pi), ib->GetDtype(x));
  auto x_square = ib->Square(x);
  auto dx = ib->Mul((ib->Muls(dout, half_root_pi)), (ib->Exp(ib->Neg(x_square))));
  return {dx};
});

REG_BPROP_BUILDER("Pow").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &exponent = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto input_dtype_id = ib->GetDtypeId(input);
  if (input_dtype_id == kNumberTypeComplex64 || input_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Pow', gradient not support for complex type currently.";
  }
  NodePtr grad_input = nullptr;
  NodePtr grad_exponent = nullptr;
  if (input->need_compute_grad_out()) {
    auto condition = ib->Equal(exponent, ib->Tensor(0, ib->GetDtype(exponent)));
    auto exponent_minus_one = ib->SubScalar(exponent, ib->ValueByType(1.0, ib->GetDtype(exponent)), ib->Value(1));
    grad_input =
      ib->Select(condition, ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(dout->dtype()->type_id()))),
                 ib->Mul(dout, ib->Mul(exponent, ib->Pow(input, exponent_minus_one))));
  }
  if (exponent->need_compute_grad_out()) {
    auto condition = ib->LogicalAnd(ib->Equal(input, ib->Tensor(0, ib->GetDtype(input))),
                                    ib->GreaterEqual(exponent, ib->Tensor(0, ib->GetDtype(exponent))));
    grad_exponent =
      ib->Select(condition, ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(dout->dtype()->type_id()))),
                 ib->Mul(dout, (ib->Mul(out, (ib->Log(input))))));
  }
  return {BinopGradCommon(ib, input, exponent, grad_input, grad_exponent)};
});

REG_BPROP_BUILDER("PowScalarTensor").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &exponent = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);

  NodePtr dexponent{nullptr};
  auto input_ptr = input->BuildValue();
  if (input_ptr->isa<ValueAny>()) {
    auto double_input = ib->Emit("ScalarCast", {input, ib->Value(static_cast<int64_t>(kNumberTypeFloat64))});
    auto log_input = ib->Emit("ScalarLog", {double_input});
    auto grad_lambda = ib->Muls(out, log_input);
    auto cond = ib->Equal(double_input, ib->Value<double>(0));
    auto true_branch = [&](Emitter *e) -> NodePtrList {
      auto exp_positive = e->GreaterEqual(exponent, e->Tensor(0, exponent->dtype()));
      auto exponent_dtype = e->Value(static_cast<int64_t>(exponent->dtype()->type_id()));
      auto zero_tensor = e->Zeros(e->Shape(exponent), exponent_dtype);
      return {e->Select(exp_positive, zero_tensor, grad_lambda)};
    };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {grad_lambda}; };
    dexponent = ib->Conditional(cond, true_branch, false_branch);
  } else {
    double input_value = PowFetchScalarValue(input_ptr, "input", "PowScalarTensor");
    auto grad_lambda = ib->Muls(out, ib->Value<double>(std::log(input_value)));
    if (fabs(input_value) < 1e-15) {
      auto exp_positive = ib->GreaterEqualScalar(exponent, ib->ValueByType(0, ib->GetDtype(exponent)));
      auto exponent_dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(exponent)));
      auto zero_tensor = ib->Zeros(ib->Shape(exponent), exponent_dtype);
      dexponent = ib->Select(exp_positive, zero_tensor, grad_lambda);
    } else {
      dexponent = grad_lambda;
    }
  }
  dexponent = ib->Mul(dout, dexponent);

  return {ib->OutZeros(input), dexponent};
});

REG_BPROP_BUILDER("PowTensorScalar").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &exponent = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  NodePtr grad_input{nullptr};
  auto exponent_ptr = exponent->BuildValue();
  if (exponent_ptr->isa<ValueAny>()) {
    auto double_exponent = ib->Emit("ScalarCast", {exponent, ib->Value(static_cast<int64_t>(kNumberTypeFloat64))});
    auto cond = ib->Equal(double_exponent, ib->Value<double>(0));
    auto true_branch = [&](Emitter *e) -> NodePtrList {
      auto input_dtype = e->Value(static_cast<int64_t>(input_x->dtype()->type_id()));
      auto grad_input = e->Zeros(e->Shape(input_x), input_dtype);
      return {grad_input};
    };
    auto false_branch = [&](Emitter *e) -> NodePtrList {
      // grad_input = grad * (exp * input.pow(exp - 1))
      auto y = e->PowTensorScalar(input_x, e->Emit("ScalarSub", {double_exponent, ib->Value<double>(1)}));
      auto grad_input = e->Mul(dout, e->Muls(y, exponent));
      return {grad_input};
    };
    grad_input = ib->Conditional(cond, true_branch, false_branch);
  } else {
    double exp_value = PowFetchScalarValue(exponent_ptr, "exponent", "PowTensorScalar");
    if (fabs(exp_value) < 1e-15) {
      auto input_dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(input_x)));
      grad_input = ib->Zeros(ib->Shape(input_x), input_dtype);
    } else {
      auto y = ib->PowTensorScalar(input_x, ib->Value<double>(exp_value - 1));
      grad_input = ib->Mul(dout, ib->Muls(y, ib->Value<double>(exp_value)));
    }
  }

  return {grad_input, ib->OutZeros(exponent)};
});

REG_BPROP_BUILDER("Exp").FreeUselessValues_I({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  TypeId exp_type = ib->GetDtypeId(out);
  if (exp_type == kNumberTypeComplex64 || exp_type == kNumberTypeComplex128) {
    out = ib->Conj(out);
  }
  auto dx = ib->Mul(dout, out);
  return {dx};
});

REG_BPROP_BUILDER("InplaceExp").FreeUselessValues_I({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  TypeId exp_type = ib->GetDtypeId(out);
  if (exp_type == kNumberTypeComplex64 || exp_type == kNumberTypeComplex128) {
    out = ib->Conj(out);
  }
  auto dx = ib->Mul(dout, out);
  return {dx};
});

REG_BPROP_BUILDER("Expm1").FreeUselessValues_I({}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  TypeId exp_type = ib->GetDtypeId(out);
  if (exp_type == kNumberTypeComplex64 || exp_type == kNumberTypeComplex128) {
    out = ib->Conj(out);
  }
  auto out_1p = ib->AddScalar(out, ib->ValueByType(1, ib->GetDtype(out)), ib->Value(1));
  auto dx = ib->Mul(dout, out_1p);
  return {dx};
});

REG_BPROP_BUILDER("Exp2").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  constexpr const double LOG_2 = 0.69314718055994530942;
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);

  // dx = dout * ln2 * 2^x -> dx = dout * ln2 * out
  auto dx = ib->Mul(dout, ib->Muls(out, ib->ValueByType(LOG_2, ib->GetDtype(out))));
  return {dx};
});

REG_BPROP_BUILDER("Minimum").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return MinimumMaximumGrad(ib, x, y, dout, true);
});

REG_BPROP_BUILDER("Maximum").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return MinimumMaximumGrad(ib, x, y, dout, false);
});

REG_BPROP_BUILDER("CumSum").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &axis = ib->GetInput(i1);
  const auto &exclusive = ib->GetInput(i2);
  const auto &reverse = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  return {ib->CumSum(dout, axis, GetValue<bool>(exclusive->BuildValue()), !GetValue<bool>(reverse->BuildValue())),
          ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
});

REG_BPROP_BUILDER("CumsumExt").FreeUselessValues_IO({i0, i2}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->GetShape(x);
  auto num_elements =
    std::accumulate(x_shape.begin(), x_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  const auto &dim = ib->GetInput(i1);
  const auto &dtype = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dim_value_ptr = dim->BuildValue();
  auto dim_opt = mindspore::GetScalarValue<int64_t>(dim_value_ptr);
  auto is_dynamic_shape = IsDynamic(x_shape);
  int64_t dim_value;
  if (dim_opt.has_value() && !is_dynamic_shape) {
    dim_value = dim_opt.value();
    auto rank = SizeToLong(x_shape.size());
    if (rank == 0) {
      rank = 1;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(dim_value >= -rank && dim_value <= rank - 1,
                               "For " + ib->name() + " grad, got an invalid dim: " + std::to_string(dim_value));
    if (dim_value < 0) {
      dim_value += rank;
    }
    if (x_shape.size() == 0 || x_shape[dim_value] == 1) {
      return {dout, ib->OutZeros(dim), ib->OutZeros(dtype)};
    }
  }
  if (!is_dynamic_shape && (num_elements <= 1)) {
    return {dout, ib->OutZeros(dim), ib->OutZeros(dtype)};
  }
  auto flip = ib->ReverseV2(dout, ib->MakeTuple({dim}));
  auto cumsum = ib->CumsumExt(flip, dim, dtype);
  auto ret = ib->ReverseV2(cumsum, ib->MakeTuple({dim}));
  return {ret, ib->OutZeros(dim), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("Cummax").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) { return CumMaxMinGrad(ib); });

REG_BPROP_BUILDER("Cummin").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) { return CumMaxMinGrad(ib); });

REG_BPROP_BUILDER("CumminExt").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) { return CumMaxMinGrad(ib); });

REG_BPROP_BUILDER("MulNoNan").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_shape = ib->Shape(x);
  auto y_shape = ib->Shape(y);
  auto dx = ib->MulNoNan(dout, y);
  auto dy = ib->MulNoNan(x, dout);
  auto bc_axis = ib->BroadcastGradientArgs(x, y);
  auto broadcast_x = bc_axis[i0];
  auto broadcast_y = bc_axis[i1];
  dx = x->need_compute_grad_out() ? ib->Reshape(ib->ReduceSum(dx, broadcast_x, false, true), x_shape) : ib->OutZeros(x);
  dy = y->need_compute_grad_out() ? ib->Reshape(ib->ReduceSum(dy, broadcast_y, false, true), y_shape) : ib->OutZeros(y);
  return {dx, dy};
});

REG_BPROP_BUILDER("BesselI0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto bessel_i1 = ib->Emit("BesselI1", {x});
  auto dx = ib->Mul(dout, bessel_i1);
  return {dx};
});

REG_BPROP_BUILDER("BesselI1").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_i0 = ib->Emit("BesselI0", {x});
  auto zero = ib->ZerosLike(x);
  auto one = ib->Fill(1.0, ib->Shape(x), ib->GetDtype(x)->type_id());
  auto dout_dx = ib->Select(ib->Equal(x, zero), one, ib->Sub(bessel_i0, (ib->Div(out, x))));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto bessel_j1 = ib->Emit("BesselJ1", {x});
  auto dx = ib->Mul(ib->Neg(dout), bessel_j1);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ1").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_j0 = ib->Emit("BesselJ0", {x});
  auto zero = ib->ZerosLike(x);
  auto zero_p5 = ib->Fill(0.5, ib->Shape(x), ib->GetDtype(x)->type_id());
  auto dout_dx = ib->Select(ib->Equal(x, zero), zero_p5, ib->Sub(bessel_j0, (ib->Div(out, x))));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto bessel_k1 = ib->Emit("BesselK1", {x});
  auto dx = ib->Mul(ib->Neg(dout), bessel_k1);
  return {dx};
});

REG_BPROP_BUILDER("BesselK1").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_k0 = ib->Emit("BesselK0", {x});
  auto dout_dx = ib->Neg(ib->Add(bessel_k0, (ib->Div(out, x))));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0e").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_k1e = ib->Emit("BesselK1e", {x});
  auto dx = ib->Mul(dout, (ib->Sub(out, bessel_k1e)));
  return {dx};
});

REG_BPROP_BUILDER("BesselK1e").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_k0e = ib->Emit("BesselK0e", {x});
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto dout_dx = ib->Sub((ib->Mul(out, (ib->Sub(one, (ib->Reciprocal(x)))))), bessel_k0e);
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselY0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto bessel_y1 = ib->Emit("BesselY1", {x});
  auto dx = ib->Mul(ib->Neg(dout), bessel_y1);
  return {dx};
});

REG_BPROP_BUILDER("BesselY1").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto bessel_y0 = ib->Emit("BesselY0", {x});
  auto dout_dx = ib->Sub(bessel_y0, (ib->Div(out, x)));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("AddN").SetUnusedInputs({i0, i1}).SetBody(AddnGradFunc);

REG_BPROP_BUILDER("AccumulateNV2").SetUnusedInputs({i0, i1}).SetBody(AddnGradFunc);

REG_BPROP_BUILDER("Tan").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Tan', gradient not support for complex type currently.";
  } else {
    dx = dout * ib->AddScalar(ib->Square(out), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1));
  }
  return {dx};
});

REG_BPROP_BUILDER("BesselI1e").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype = ib->GetDtype(x);
  auto zeros = ib->ZerosLike(x);
  auto eps = GetEps(ib, x_dtype);
  auto x_is_valid = ib->Less(eps, ib->Abs(x));
  auto x_safe = ib->Select(x_is_valid, x, eps + zeros);
  auto besselI0e = ib->Emit(prim::kPrimBesselI0e->name(), {x_safe});
  auto tmp = besselI0e - out * (ib->Sign(x_safe) + ib->Reciprocal(x_safe));
  auto dx = ib->Select(x_is_valid, tmp, ib->Tensor(0.5, x_dtype) + zeros) * dout;
  return {dx};
});

REG_BPROP_BUILDER("Atanh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_dtype = ib->GetDtype(x);
  auto x_dtype_id = x_dtype->type_id();
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Atanh', gradient not support for complex type currently.";
  } else {
    dx = ib->Div(dout, ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  }
  return {dx};
});

REG_BPROP_BUILDER("Inv").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit(prim::kPrimInvGrad->name(), {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("LinSpace").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("LinSpaceExt").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("DynamicQuantExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IndexAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  auto axis = ib->EmitValue(ib->GetAttr(kAttrAxis));
  auto dy = ib->Gather(dout, indices, axis);
  return {dout, ib->OutZeros(indices), dy};
});

REG_BPROP_BUILDER("InplaceIndexPut").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &indices = ib->GetInput(i1);
  const auto &values = ib->GetInput(i2);
  const auto &accumulate = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  // Indices is tuple[tensor]
  std::vector<ShapeVector> indices_shapes = indices->shapes();
  auto indices_nums = indices_shapes.size();
  NodePtrList indices_res;
  for (size_t i = 0; i < indices_nums; ++i) {
    indices_res.push_back(ib->OutZeros(ib->TupleGetItem(indices, i)));
  }
  NodePtr values_grad = nullptr;
  if (values->need_compute_grad_out()) {
    values_grad = ib->Index(dout, indices);
    auto values_shape = ib->Shape(values);
    auto values_grad_shape = ib->Shape(values_grad);
    auto bc_values = ib->BroadcastGradientArgs(values, values_grad)[i0];
    auto bc_values_shape_ptr = bc_values->GetShape();
    // Dynamic if op need control flow
    if (bc_values_shape_ptr->isa<abstract::DynamicSequenceShape>()) {
      // values_grad shape=(1, ....) don't use sum op
      auto true_branch = [&values_grad, &values_shape](Emitter *e) -> NodePtrList {
        return {e->Reshape(values_grad, values_shape)};
      };
      auto false_branch = [&values_grad, &values_shape, &bc_values](Emitter *e) -> NodePtrList {
        auto sum_ext_result = e->SumExt(values_grad, bc_values, e->Value(false), e->EmitValue(kNone));
        return {e->Reshape(sum_ext_result, values_shape)};
      };
      auto cond = ib->Equal(ib->Emit("sequence_len", {bc_values}), ib->Value<int64_t>(0));
      values_grad = ib->Conditional(cond, true_branch, false_branch);
    } else {
      auto bc_values_shape = bc_values_shape_ptr->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(bc_values_shape);
      if (bc_values_shape->size() > 0) {
        values_grad =
          ib->Reshape(ib->SumExt(values_grad, bc_values, ib->Value(false), ib->EmitValue(kNone)), values_shape);
      } else {
        values_grad = ib->Reshape(values_grad, values_shape);
      }
    }
  } else {
    values_grad = ib->OutZeros(values);
  }
  NodePtr dx = nullptr;
  if (input->need_compute_grad_out()) {
    auto accumulate_opt = mindspore::GetScalarValue<bool>(accumulate->BuildValue());
    if (accumulate_opt.has_value()) {
      auto dout_clone = ib->Clone(dout);
      if (!accumulate_opt.value()) {
        auto zeros = ib->Zeros(ib->Shape(values), ib->Value(static_cast<int64_t>(ib->GetDtypeId(values))));
        dx = ib->InplaceIndexPut(dout_clone, indices, zeros, accumulate);
      } else {
        dx = dout_clone;
      }
    } else {
      auto accumulate_false_branch = [&dout, &values, &indices, &accumulate](Emitter *e) -> NodePtrList {
        auto dout_clone = e->Clone(dout);
        auto values_dtype = e->Value(static_cast<int64_t>(values->dtype()->type_id()));
        auto zeros = e->Zeros(e->Shape(values), values_dtype);
        auto dx_r = e->InplaceIndexPut(dout_clone, indices, zeros, accumulate);
        return {dx_r};
      };
      auto accumulate_true_branch = [&dout](Emitter *e) -> NodePtrList {
        auto dout_clone = e->Clone(dout);
        return {dout_clone};
      };
      dx = ib->Conditional(accumulate, accumulate_true_branch, accumulate_false_branch);
    }
  } else {
    dx = ib->OutZeros(input);
  }
  return {dx, ib->MakeTuple(indices_res), values_grad, ib->OutZeros(accumulate)};
});

REG_BPROP_BUILDER("IndexSelect").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &index = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto zeros = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  auto dx = ib->InplaceIndexAddExt(zeros, axis, index, dout, ib->Value<int64_t>(1LL));
  return {dx, ib->OutZeros(axis), ib->OutZeros(index)};
});

REG_BPROP_BUILDER("Logit").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &eps = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("LogitGrad", {dout, x, eps});
  return {dx, ib->OutZeros(eps)};
});

DEF_PURE_SHAPE_CALC(g_cdist)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto dout_shape = inputs.at(0);
    auto dout_dim = dout_shape.size();
    ShapeVector perm;
    for (uint64_t i = 0; i < dout_dim - 2; ++i) {
      perm.push_back(i);
    }
    perm.push_back(dout_dim - 1);
    perm.push_back(dout_dim - 2);
    return {perm};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto dout_shape = inputs.at(0);
    return {IsDynamicRank(dout_shape) ? -1 : static_cast<int64_t>(dout_shape.size())};
  });
REG_BPROP_BUILDER("Cdist").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &input_y = ib->GetInput(i1);
  const auto &p = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto res = ib->ShapeCalc(g_cdist, {dout})[0];
  auto dout_transpose = ib->Transpose(dout, res);
  auto out_transpose = ib->Transpose(out, res);
  auto dx =
    input_x->need_compute_grad_out() ? ib->Emit("CdistGrad", {dout, input_x, input_y, out, p}) : ib->OutZeros(input_x);
  auto dy = input_y->need_compute_grad_out()
              ? ib->Emit("CdistGrad", {dout_transpose, input_y, input_x, out_transpose, p})
              : ib->OutZeros(input_y);
  return {dx, dy, ib->OutZeros(p)};
});

REG_BPROP_BUILDER("LuUnpack").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &lu_data = ib->GetInput(i0);
  const auto &lu_pivots = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto tmp = ib->Emit(
    "LuUnpackGrad", {ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2), lu_data},
    {{"L_grad_flag", MakeValue(true)}, {"U_grad_flag", MakeValue(true)}, {"cust_aicpu", MakeValue("LuUnpackGrad")}});
  auto dl = ib->TupleGetItem(tmp, 0);
  auto du = ib->TupleGetItem(tmp, 1);
  auto lu_data_grad = ib->Add(dl, du);
  return {lu_data_grad, ib->OutZeros(lu_pivots)};
});

REG_BPROP_BUILDER("Sinc").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto product = ib->Muls(x, ib->ValueByType(pi, ib->GetDtype(x)));
  auto dx = ib->Div((ib->Mul((ib->Sub((ib->Cos(product)), out)), dout)), x);
  TypeId x_type = ib->GetDtypeId(x);
  if (x_type == kNumberTypeComplex64 || x_type == kNumberTypeComplex128) {
    dx = ib->Conj(dx);
  }
  auto zeros = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  auto cond = ib->Equal(product, ib->Tensor(0.0, ib->GetDtype(x)));
  return {ib->Select(cond, zeros, dx)};
});

DEF_PURE_SHAPE_CALC(g_cumprod_axis2tuple)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(i0);
    auto axis = inputs.at(i1)[0];

    auto normal_axis = NormalizeAxis(axis, x_shape.size());
    std::vector<int64_t> res;
    res.push_back(x_shape[normal_axis]);
    return {res};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

REG_BPROP_BUILDER("CumProd").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->GetShape(x);
  const auto &axis = ib->GetInput(i1);
  const auto &exclusive = ib->GetInput(i2);
  const auto &reverse = ib->GetInput(i3);
  auto out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto axis_value_ptr = axis->BuildValue();
  auto axis_opt = mindspore::GetScalarValue<int64_t>(axis_value_ptr);
  // dynamic axis
  if (!axis_opt.has_value()) {
    auto axis_one_branch = [&](Emitter *e) -> NodePtrList { return {dout}; };
    auto axis_not_one_branch = [&](Emitter *e) -> NodePtrList {
      auto prod = e->CumProd(x, axis, exclusive, reverse);
      out = e->CumSum(e->Mul(prod, dout), axis, GetValue<bool>(exclusive->BuildValue()),
                      !GetValue<bool>(reverse->BuildValue()));
      out = e->RealDiv(out, x);
      return {out};
    };
    auto axis_to_tuple = ib->ShapeCalc(g_cumprod_axis2tuple, {x, axis}, {1})[0];
    auto cond = ib->Equal(ib->TupleGetItem(axis_to_tuple, 0), ib->Value<int64_t>(1));
    out = ib->Conditional(cond, axis_one_branch, axis_not_one_branch);
    return {out, ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
  }
  int64_t axis_value = axis_opt.value();
  constexpr const int64_t One = 1;
  // to par with standards when dim is 1 or element num of input is no greater than 1.
  if (!IsDynamic(x_shape) && x_shape[axis_value] == One) {
    return {dout, ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
  }
  auto prod = ib->CumProd(x, axis, exclusive, reverse);
  out = ib->CumSum(ib->Mul(prod, dout), axis, GetValue<bool>(exclusive->BuildValue()),
                   !GetValue<bool>(reverse->BuildValue()));
  return {ib->RealDiv(out, x), ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
});

REG_BPROP_BUILDER("IsFinite").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsNan").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsInf").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAll").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAny").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ApproximateEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Equal").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("EqScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("EqualExt").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NotEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Greater").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("GreaterEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("GreaterEqualScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("MatrixInverse").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto out_shape = ib->GetShape(out);
  auto dx = out;
  if (out_shape.size() == 2) {
    dx = ib->MatMul(dout, dx, false, true);
    dx = ib->MatMul(out, dx, true, false);
  } else if (out_shape.size() > 2 || IsDynamicRank(out_shape)) {
    dx = ib->BatchMatMul(dout, dx, false, true);
    dx = ib->BatchMatMul(out, dx, true, false);
  }
  return {-dx};
});

REG_BPROP_BUILDER("MatrixInverseExt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto out_shape = ib->GetShape(out);
  auto dx = out;
  if (out_shape.size() == 2) {
    dx = ib->MatMul(dout, dx, false, true);
    dx = ib->MatMul(out, dx, true, false);
  } else if (out_shape.size() > 2 || IsDynamicRank(out_shape)) {
    dx = ib->BatchMatMul(dout, dx, false, true);
    dx = ib->BatchMatMul(out, dx, true, false);
  }
  return {-dx};
});

REG_BPROP_BUILDER("Neg").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {-dout};
});

REG_BPROP_BUILDER("RealDiv").FreeUselessValues(FreeTensorsOfDiv).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'RealDiv', gradient not support for complex type currently.";
  }
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto bc_dx = ib->RealDiv(dout, y);
  NodePtr bc_dy = nullptr;
  if (y->need_compute_grad_out()) {
    bc_dy = -(bc_dx * out);
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("DivNoNan").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto bc_dx = ib->DivNoNan(dout, y);
  NodePtr bc_dy = nullptr;
  if (y->need_compute_grad_out()) {
    bc_dy = -(bc_dx * out);
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Xdivy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
    bc_dx = (ib->Xdivy(not_zero_x, y)) * dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (ib->Xdivy(-x, ib->Square(y))) * dout;
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("FloorDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("FloorDivScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceFloorDivide").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceFloorDivides").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("FloorMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Cast(dout, ib->GetDtype(x));
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->FloorDiv(x, y));
    bc_dy = ib->Cast(bc_dy, ib->GetDtype(y));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("RemainderTensorScalar").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  auto dout = ib->Cast(ib->GetInput(i3), ib->GetDtype(input));
  return {dout, ib->OutZeros(other)};
});

REG_BPROP_BUILDER("RemainderTensorTensor").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr d_input = dout;
  NodePtr d_other = nullptr;
  if (other->need_compute_grad_out()) {
    d_other = (-dout) * (ib->DivMod(input, other, ops::RoundingMode::FLOOR));
  }
  const auto &grads = BinopGradCommon(ib, input, other, d_input, d_other);
  d_input = input->need_compute_grad_out() ? ib->Cast(grads[i0], ib->GetDtype(input)) : nullptr;
  d_other = other->need_compute_grad_out() ? ib->Cast(grads[i1], ib->GetDtype(other)) : nullptr;
  return {d_input, d_other};
});

REG_BPROP_BUILDER("RemainderScalarTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("InplaceRemainderTensorScalar").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(other)};
});

inline static bool CloneInplaceInputFuncForInplaceRemainderTensorTensor(const PynativeCallback &cb) {
  if (!cb.IsNotRequiresGrad(i1)) {
    return true;
  }
  return false;
}

static void FreeTensorsOfInplaceRemainderTensorTensor(const PynativeCallback &cb) {
  cb.FreeOutputDeviceAddress();
  auto &inputs = *cb.GetInputs();
  if (cb.IsNotRequiresGrad(i1)) {
    cb.FreeDeviceAddress(&inputs[i0]);
    cb.FreeDeviceAddress(&inputs[i1]);
    MS_LOG(DEBUG) << "Clear device address for inputs[0] and inputs[1] of " << cb.opname();
  }
}

REG_BPROP_BUILDER("InplaceRemainderTensorTensor")
  .SetUnusedInputs({i2})
  .CloneInplaceInput(CloneInplaceInputFuncForInplaceRemainderTensorTensor)
  .FreeUselessValues(FreeTensorsOfInplaceRemainderTensorTensor)
  .SetBody(BODYFUNC(ib) {
    auto input = ib->GetInput(i0);
    auto other = ib->GetInput(i1);
    auto dout = ib->GetInput(i3);
    NodePtr d_input = input->need_compute_grad_out() ? dout : nullptr;
    NodePtr d_other = nullptr;
    if (other->need_compute_grad_out()) {
      d_other = (-dout) * (ib->DivMod(input, other, ops::RoundingMode::FLOOR));
      d_other = ib->Cast(d_other, ib->GetDtype(other));
    }
    return {BinopGradCommon(ib, input, other, d_input, d_other)};
  });

REG_BPROP_BUILDER("TruncateDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("TruncateMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->Emit("TruncateDiv", {x, y}));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Mod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Cast(dout, ib->GetDtype(x));
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->FloorDiv(x, y));
    bc_dy = ib->Cast(bc_dy, ib->GetDtype(y));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Xlogy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    // dx = xlog(grad, other).masked_fill((self == 0) & (other <= 0.), 0.)
    auto zero = ib->Tensor(0.0, x_dtype);
    auto input = ib->Xlogy(dout, y);
    auto mask = ib->BitwiseAndTensor(ib->Equal(x, zero), ib->LessEqual(y, zero));
    bc_dx = ib->MaskedFill(input, mask, zero);
  }
  if (y->need_compute_grad_out()) {
    // dy = grad * self / other
    bc_dy = ib->Div(ib->Mul(dout, x), y);
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("XLogYScalarSelf").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  // input, other, out, dout
  auto x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = ib->OutZeros(x);
  x = ib->ScalarToTensor(x, ib->GetDtype(y));
  NodePtr bc_dy = ib->Div(x, y) * dout;
  return {bc_dx, bc_dy};
});

REG_BPROP_BUILDER("XLogYScalarOther").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  // input, other, out, dout
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_dtype = ib->GetDtype(x);
  auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
  NodePtr bc_dx = ib->XLogYScalarOther(not_zero_x, y) * dout;
  NodePtr bc_dy = ib->OutZeros(y);
  return {bc_dx, bc_dy};
});

REG_BPROP_BUILDER("Sqrt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("SqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("SqrtGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto gy = ib->RealDiv(dout, y);
  auto dy = y->need_compute_grad_out() ? (-gy) * out : ib->OutZeros(y);
  NodePtr dgrad;
  if (grad->need_compute_grad_out()) {
    auto gy_dtype = ib->GetDtype(gy);
    dgrad = ib->Tensor(0.5, gy_dtype) * gy;
  } else {
    dgrad = ib->OutZeros(grad);
  }
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Rsqrt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("RsqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("RsqrtGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto grad_dtype = ib->GetDtype(grad);
    dy = ib->Tensor(-1.5, grad_dtype) * grad * y * y * dout;
  } else {
    dy = ib->OutZeros(y);
  }
  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("RsqrtGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Reciprocal").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dtype = ib->GetDtypeId(out);
  bool is_complex = (dtype == kNumberTypeComplex64 || dtype == kNumberTypeComplex128);
  return {is_complex ? -dout * ib->Conj(out * out) : -dout * (out * out)};
});

REG_BPROP_BUILDER("Log").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  return {ib->Div(dout, x)};
});

REG_BPROP_BUILDER("InplaceLog").CloneInplaceInput().SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  return {ib->Div(dout, x)};
});

REG_BPROP_BUILDER("Log2").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  double denominator = 0.6931471805599453;
  return {ib->Div(dout, ib->Muls(x, ib->Value<double>(denominator)))};
});

REG_BPROP_BUILDER("Log10").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  double denominator = 2.3025850929940456;
  return {ib->Div(dout, ib->Muls(x, ib->Value<double>(denominator)))};
});

REG_BPROP_BUILDER("Floor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto zeros = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {zeros};
});

REG_BPROP_BUILDER("InplaceFloor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto zeros = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {zeros};
});

REG_BPROP_BUILDER("Ceil").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Square").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = dout * ib->Muls(ib->PowTensorScalar(x, ib->Value<int64_t>(1)), ib->Value<int64_t>(2));
  return {dx};
});

REG_BPROP_BUILDER("SquaredDifference").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Muls(dout * (x - y), ib->ValueByType(2.0, ib->GetDtype(x)));
  return {BinopGradCommon(ib, x, y, dx, -dx)};
});

REG_BPROP_BUILDER("SquareSumAll").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx;
  if (x->need_compute_grad_out()) {
    auto dout_0 = ib->TupleGetItem(dout, i0);
    dx = ib->Muls(dout_0 * x, ib->ValueByType(2.0, ib->GetDtype(x)));
  } else {
    dx = ib->OutZeros(x);
  }
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto dout_1 = ib->TupleGetItem(dout, i1);
    dy = ib->Muls(dout_1 * y, ib->ValueByType(2.0, ib->GetDtype(y)));
  } else {
    dy = ib->OutZeros(y);
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("Hypot").SetBody(BODYFUNC(ib) {
  const auto &x1 = ib->GetInput(i0);
  const auto &x2 = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto x1_f32 = ib->Cast(x1, kFloat32);
  auto x2_f32 = ib->Cast(x2, kFloat32);
  auto out_f32 = ib->Cast(out, kFloat32);
  auto dout_f32 = ib->Cast(dout, kFloat32);
  NodePtr dx1 = nullptr;
  NodePtr dx2 = nullptr;
  if (x1->need_compute_grad_out()) {
    dx1 = ib->Mul(ib->Div(x1_f32, out_f32), dout_f32);
  }
  if (x2->need_compute_grad_out()) {
    dx2 = ib->Mul(ib->Div(x2_f32, out_f32), dout_f32);
  }
  auto tmp = BinopGradCommon(ib, x1_f32, x2_f32, dx1, dx2);
  auto result_dx1 = x1->need_compute_grad_out() ? ib->Cast(tmp[0], ib->GetDtype(x1)) : ib->OutZeros(x1);
  auto result_dx2 = x2->need_compute_grad_out() ? ib->Cast(tmp[1], ib->GetDtype(x2)) : ib->OutZeros(x2);
  return {result_dx1, result_dx2};
});

REG_BPROP_BUILDER("Trunc").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto zeros = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {zeros};
});

REG_BPROP_BUILDER("Ger").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &input_y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  ShapeVector axis = {1};
  NodePtr dx;
  if (input_x->need_compute_grad_out()) {
    auto m1 = ib->ExpandDims(input_y, 1);
    dx = ib->Squeeze(ib->MatMul(dout, m1, false, false), MakeValue(axis));
  } else {
    dx = ib->OutZeros(input_x);
  }
  NodePtr dy;
  if (input_y->need_compute_grad_out()) {
    auto m2 = ib->ExpandDims(input_x, 1);
    ShapeVector perm = {1, 0};
    auto transpose = ib->Transpose(dout, perm);
    dy = ib->Squeeze(ib->MatMul(transpose, m2, false, false), MakeValue(axis));
  } else {
    dy = ib->OutZeros(input_y);
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("Cross").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto other = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  auto input_type_id = ib->GetDtypeId(input);
  bool is_complex = (input_type_id == kNumberTypeComplex64 || input_type_id == kNumberTypeComplex128);
  if (input->need_compute_grad_out() && is_complex) {
    other = ib->Conj(ib->GetInput(i1));
  }
  if (other->need_compute_grad_out() && is_complex) {
    input = ib->Conj(ib->GetInput(i0));
  }
  auto dinput = input->need_compute_grad_out() ? ib->Cross(other, dout, dim) : ib->OutZeros(input);
  auto dother = other->need_compute_grad_out() ? ib->Cross(dout, input, dim) : ib->OutZeros(other);
  std::vector<NodePtr> ret = BinopGradCommon(ib, input, other, dinput, dother);
  ret.emplace_back(ib->OutZeros(dim));
  return ret;
});

REG_BPROP_BUILDER("Median").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx =
    ib->Cast(ib->Emit("MedianGrad", {ib->TupleGetItem(dout, 0), x, ib->TupleGetItem(out, 0), ib->TupleGetItem(out, 1)},
                      {{"global_median", ib->GetAttr("global_median")},
                       {"axis", ib->GetAttr("axis")},
                       {"keep_dims", ib->GetAttr("keep_dims")}}),
             ib->GetDtype(x));
  return {dx};
});

REG_BPROP_BUILDER("MedianExt").SetBody(BODYFUNC(ib) {
  auto dx = ReduceExtOpGrad(ib, ib->GetInput(i0), ib->GetInput(i1), ib->GetInput(i2));
  return {dx};
});
REG_BPROP_BUILDER("Max").SetBody(BODYFUNC(ib) {
  auto dx = ReduceExtOpGrad(ib, ib->GetInput(i0), ib->GetInput(i1), ib->GetInput(i2));
  return {dx};
});
REG_BPROP_BUILDER("Min").SetBody(BODYFUNC(ib) {
  auto dx = ReduceExtOpGrad(ib, ib->GetInput(i0), ib->GetInput(i1), ib->GetInput(i2));
  return {dx};
});

REG_BPROP_BUILDER("MedianDim").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto dx = MeidanDimGrad(ib, x, axis, keep_dims, out, dout);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("Trace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto shape = ib->Shape(x, true);
  auto dx = ib->Emit("TraceGrad", {dout, shape});
  return {dx};
});

REG_BPROP_BUILDER("TraceV2").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &offset = ib->GetInput(i1);
  const auto &axis1 = ib->GetInput(i2);
  const auto &axis2 = ib->GetInput(i3);
  const auto &dtype = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  auto shape = ib->Shape(input, true);
  auto dx = ib->Emit("TraceV2Grad", {dout, shape, offset, axis1, axis2});
  return {dx, ib->OutZeros(offset), ib->OutZeros(axis1), ib->OutZeros(axis2), ib->OutZeros(dtype)};
});

DEF_PURE_SHAPE_CALC(g_trace_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(i0);
    return {input_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    if (IsDynamicRank(inputs.at(i0))) {
      return {-1};
    }
    return {SizeToLong(inputs.at(i0).size())};
  });

REG_BPROP_BUILDER("TraceExt").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto shape = ib->GetShape(x);
  auto dtype_id = ib->GetDtypeId(out);
  NodePtr eye = nullptr;
  TypeId eye_dtype_id = kTypeUnknown;
  if (dtype_id == kNumberTypeBFloat16) {
    eye_dtype_id = kNumberTypeFloat32;
  } else {
    eye_dtype_id = dtype_id;
  }
  if (IsDynamicShape(shape) || IsDynamicRank(shape)) {
    auto shapes = ib->ShapeCalc(g_trace_ext_shapecalc, {x});
    eye = ib->Eye(ib->TupleGetItem(shapes[0], 0), ib->TupleGetItem(shapes[0], 1), ib->Value<int64_t>(eye_dtype_id));
  } else {
    eye = ib->Eye(ib->Value(shape[0]), ib->Value(shape[1]), ib->Value<int64_t>(eye_dtype_id));
  }
  auto dx = ib->Mul(eye, dout);
  if (dtype_id == kNumberTypeBFloat16) {
    dx = ib->Cast(dx, kBFloat16);
    return {dx};
  }
  return {dx};
});

REG_BPROP_BUILDER("Erfinv").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_type = ib->GetDtype(x);
  auto pow = ib->Exp(ib->PowTensorScalar(out, ib->Value(2)));
  auto dx = ib->Mul(ib->Muls(pow, ib->ValueByType(sqrt(pi) / 2, x_type)), dout);
  return {dx};
});

inline static bool CloneInplaceInputFuncForInplaceErfinv(const PynativeCallback &cb) {
  if (!cb.IsNotRequiresGrad(i0)) {
    return true;
  }
  return false;
}

REG_BPROP_BUILDER("InplaceErfinv").CloneInplaceInput(CloneInplaceInputFuncForInplaceErfinv).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_type = ib->GetDtype(x);
  auto exp_self_erfinv_pow_2 = ib->Exp(ib->PowTensorScalar(out, ib->Value(2)));
  auto dx = ib->Mul(ib->Muls(exp_self_erfinv_pow_2, ib->ValueByType(sqrt(pi) / 2, x_type)), dout);
  return {dx};
});

REG_BPROP_BUILDER("Bernoulli").FreeUselessValues_IO({}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("BernoulliExt").FreeUselessValues_IO({i0, i1, i2}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &seed = ib->GetInput(i1);
  const auto &offset = ib->GetInput(i2);
  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, ib->OutZeros(seed), ib->OutZeros(offset)};
});

NodePtrList InplaceBernoulliGrad(BpropBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto p = ib->GetInput(kIndex1);
  auto seed = ib->GetInput(kIndex2);
  auto offset = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    dx = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  }
  return {dx, ib->OutZeros(p), ib->OutZeros(seed), ib->OutZeros(offset)};
}
REG_BPROP_BUILDER("InplaceBernoulliTensor").FreeUselessValues_IO({i0, i1, i2, i3}, {}).SetBody(InplaceBernoulliGrad);
REG_BPROP_BUILDER("InplaceBernoulliScalar").FreeUselessValues_IO({i0, i1, i2, i3}, {}).SetBody(InplaceBernoulliGrad);

REG_BPROP_BUILDER("ReduceSum").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &skip_mode = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx =
    SumGrad(ib, x, axis, dout, GetValue<bool>(keep_dims->BuildValue()), GetValue<bool>(skip_mode->BuildValue()));
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(skip_mode)};
});

DEF_PURE_SHAPE_CALC(g_reduce_prod)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(0);
    auto axis = inputs.at(1);
    auto output_shape_kept_dims = ReduceShape(input_shape, axis);
    auto tile_scaling = ReduceShapeTupleDiv(input_shape, output_shape_kept_dims);
    auto [pack_shape, perm] = SplitShapeIndex(input_shape, axis);
    return {output_shape_kept_dims, tile_scaling, pack_shape, perm, InvertPermutation(perm)};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto input_shape = inputs.at(0);
    if (IsDynamicRank(input_shape) || !unknown_inputs.empty()) {
      return {-1, -1, 2, -1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, size, 2, size, size};
  });

REG_BPROP_BUILDER("ReduceProd").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceProd', gradient not support for complex type currently.";
  }
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  if (ib->GetRank(x) == 0) {
    return {dout, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
  auto res = ib->ShapeCalc(g_reduce_prod, {x, axis}, {1});
  auto keep_dims_value = GetValue<bool>(keep_dims->BuildValue());
  auto grad = keep_dims_value ? dout : ib->Reshape(dout, res[0]);
  grad = ib->Tile(grad, res[1]);

  auto permuted = ib->Transpose(x, res[3]);
  auto permuted_shape = ib->Shape(permuted);
  auto reshaped = ib->Reshape(permuted, res[2]);
  auto left = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, false);
  auto right = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, true);
  auto y = ib->Reshape(ib->Mul(left, right), permuted_shape);
  auto out = ib->Mul(ib->Transpose(y, res[4]), grad);
  auto dx = ib->Reshape(out, ib->Shape(x));
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ReduceMax").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMax', gradient not support for complex type currently.";
  } else {
    NodePtr dx;
    auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
    if (!keep_dims_opt.has_value()) {
      auto true_branch = [&](Emitter *e) -> NodePtrList { return {MinOrMaxGrad(e, x, axis, true, out, dout)}; };
      auto false_branch = [&](Emitter *e) -> NodePtrList { return {MinOrMaxGrad(e, x, axis, false, out, dout)}; };
      auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
      dx = ib->Conditional(keep_dims_true, true_branch, false_branch);
    } else {
      dx = MinOrMaxGrad(ib, x, axis, keep_dims_opt.value(), out, dout);
    }
    return {ib->Cast(dx, ib->GetDtype(x)), ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
});

REG_BPROP_BUILDER("ReduceMin").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMin', gradient not support for complex type currently.";
  } else {
    NodePtr dx;
    auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
    if (!keep_dims_opt.has_value()) {
      auto true_branch = [&](Emitter *e) -> NodePtrList { return {MinOrMaxGrad(e, x, axis, true, out, dout)}; };
      auto false_branch = [&](Emitter *e) -> NodePtrList { return {MinOrMaxGrad(e, x, axis, false, out, dout)}; };
      auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
      dx = ib->Conditional(keep_dims_true, true_branch, false_branch);
    } else {
      dx = MinOrMaxGrad(ib, x, axis, keep_dims_opt.value(), out, dout);
    }
    return {ib->Cast(dx, ib->GetDtype(x)), ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
});

REG_BPROP_BUILDER("ReduceMean").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMean', gradient not support for complex type currently.";
  }
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto grad = SumGrad(ib, x, axis, dout, GetValue<bool>(keep_dims->BuildValue()));
  NodePtr div_shape_node;
  if (IsDynamic(ib->GetShape(x)) || IsDynamic(ib->GetShape(out))) {
    auto shape_out_sz = ib->DynSize(out, kFloat32);
    auto div_shape = ib->DynSize(x, kFloat32) / shape_out_sz;
    div_shape_node = ib->Cast(div_shape, ib->GetDtype(grad));
  } else {
    auto shape_out_sz = ib->GetSize(out);
    if (shape_out_sz == 0) {
      MS_EXCEPTION(ValueError) << "out shape size can not be 0";
    }
    auto div_shape = ib->GetSize(x) / shape_out_sz;
    div_shape_node = ib->Tensor(div_shape, ib->GetDtype(grad));
  }
  auto dx = ib->RealDiv(grad, div_shape_node);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ArgMaxWithValue").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, true);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ArgMinWithValue").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, false);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("MaxDim").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, true, true);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("MinDim").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, false, true);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ComplexAbs").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  return {ib->DivNoNan(ib->Mul(ib->Complex(dout, ib->ZerosLike(dout)), x), ib->Complex(out, ib->ZerosLike(out)))};
});

REG_BPROP_BUILDER("Real").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Complex(dout, zero)};
});

REG_BPROP_BUILDER("Imag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto zero = ib->ZerosLike(dout);
  auto x_dtype_id = ib->GetDtypeId(x);
  auto dx = ib->Complex(zero, dout);
  if (x_dtype_id != kNumberTypeComplex64 && x_dtype_id != kNumberTypeComplex128) {
    return {ib->Real(dx)};
  }
  return {dx};
});

REG_BPROP_BUILDER("RealView").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(i2);
  auto zero = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  return {ib->Complex(dout, zero)};
});

REG_BPROP_BUILDER("ImagView").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(i2);
  auto zero = ib->Zeros(ib->Shape(dout), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  return {ib->Complex(zero, dout)};
});

REG_BPROP_BUILDER("Betainc").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &input_a = ib->GetInput(i0);
  const auto &input_b = ib->GetInput(i1);
  const auto &input_x = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto sx = ib->Shape(input_x);
  auto log_beta =
    ib->Emit("Lgamma", {input_a}) + ib->Emit("Lgamma", {input_b}) - ib->Emit("Lgamma", {ib->Add(input_a, input_b)});
  auto partial_x = ib->Exp(ib->Sub(
    (ib->Add((ib->Mul((ib->SubScalar(input_b, ib->ValueByType(1, ib->GetDtype(input_b)), ib->Value(1))),
                      (ib->Log1p(ib->Neg(input_x))))),
             (ib->Xlogy(ib->SubScalar(input_a, ib->ValueByType(1, ib->GetDtype(input_b)), ib->Value(1)), input_x)))),
    log_beta));
  return {ib->OutZeros(input_a), ib->OutZeros(input_b), ib->Reshape(ib->Mul(partial_x, dout), sx)};
});

DEF_PURE_SHAPE_CALC(g_matrix_determinant).SetCalc(MatrixDeterminantShapeFunc).SetInfer(MatrixDeterminantInferFunc);
REG_BPROP_BUILDER("LogMatrixDeterminant").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto res = ib->ShapeCalc(g_matrix_determinant, {ib->TupleGetItem(out, 1)})[0];
  auto multipliers = ib->Reshape(ib->TupleGetItem(dout, 1), res);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixDeterminant").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto res = ib->ShapeCalc(g_matrix_determinant, {out})[0];
  auto multipliers = ib->Reshape(ib->Mul(dout, out), res);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixPower").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto n = GetValue<int64_t>(ib->GetAttr("n"));
  auto x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  dout = ib->Cast(dout, kFloat32);
  x = ib->Cast(x, kFloat32);
  auto power = n;
  auto dx = ib->ZerosLike(x);
  auto EmitBmmPowerGrad = [&ib, &dx, &dout](int64_t repeat, const NodePtr &a, const NodePtr &b) {
    for (int64_t i = 0; i < repeat; ++i) {
      dx = ib->Add(dx, (ib->BatchMatMul(b, ib->Emit("MatrixPower", {a}, {{"n", MakeValue<int64_t>(repeat - 1 - i)}}),
                                        false, true)));
      dout = ib->BatchMatMul(a, b, true, false);
    }
  };
  if (power < 0) {
    auto x_inv = ib->Emit("MatrixPower", {x}, {{"n", MakeValue<int64_t>(-1)}});
    EmitBmmPowerGrad(-power, x_inv, dout);
    dx = ib->BatchMatMul(dx, x_inv, false, true);
    dx = ib->BatchMatMul(x_inv, dx, true, false);
    dx = ib->Neg(dx);
  } else {
    EmitBmmPowerGrad(power, x, dout);
  }
  dx = ib->Cast(dx, ib->GetDtype(out));
  return {dx};
});

REG_BPROP_BUILDER("MatrixSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto adjoint = GetValue<bool>(ib->GetAttr("adjoint"));
  const auto &input_a = ib->GetInput(i0);
  auto out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  if (ib->GetDtypeId(out) == kNumberTypeFloat64) {
    out = ib->Cast(out, kFloat32);
  }
  auto grad_b = ib->Emit("MatrixSolve", {input_a, dout}, {{"adjoint", MakeValue(!adjoint)}});
  if (ib->GetDtypeId(grad_b) == kNumberTypeFloat64) {
    grad_b = ib->Cast(grad_b, kFloat32);
  }
  auto a_shape = ib->GetShape(input_a);
  auto matrix_rank = a_shape.size();
  auto EmitBmmGrad = [&ib](const NodePtr &a, const NodePtr &b) -> NodePtr {
    auto grad_a = ib->BatchMatMul(a, b, false, true);
    grad_a = ib->Neg(grad_a);
    return grad_a;
  };
  auto EmitMatmulGrad = [&ib](const NodePtr &a, const NodePtr &b) -> NodePtr {
    auto grad_a = ib->MatMul(a, b, false, true);
    grad_a = ib->Neg(grad_a);
    return grad_a;
  };
  if (adjoint) {
    if (matrix_rank > 2) {
      return {EmitBmmGrad(out, grad_b), grad_b};
    } else {
      return {EmitMatmulGrad(out, grad_b), grad_b};
    }
  } else {
    if (matrix_rank > 2) {
      return {EmitBmmGrad(grad_b, out), grad_b};
    } else {
      return {EmitMatmulGrad(grad_b, out), grad_b};
    }
  }
});

DEF_PURE_SHAPE_CALC(g_matrix_exp)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto shape_x = inputs.at(0);
    auto x_len = shape_x.size();
    auto input_perm = Range(SizeToLong(x_len));
    input_perm[x_len - i2] = SizeToLong(x_len - i1);
    input_perm[x_len - i1] = SizeToLong(x_len - i2);
    auto n = shape_x[x_len - i1];
    ShapeVector begins(x_len, 0);
    begins[x_len - i1] = n;
    auto sizes = shape_x;
    sizes[x_len - i1] = n;
    sizes[x_len - i2] = n;
    return {input_perm, begins, sizes};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : static_cast<int64_t>(shape_x.size());
    return {rank, rank, rank};
  });

REG_BPROP_BUILDER("MatrixExp").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto res = ib->ShapeCalc(g_matrix_exp, {x});
  auto input_perm = res[0];
  auto begins = res[1];
  auto sizes = res[2];
  auto x_transpose = ib->Transpose(x, input_perm);
  auto zero_matrix = ib->ZerosLike(x);
  zero_matrix = ib->Cast(zero_matrix, ib->GetDtype(dout));
  auto meta_grad_up = ib->Concat({x_transpose, dout}, -1);
  auto meta_grad_down = ib->Concat({zero_matrix, x_transpose}, -1);
  auto meta_grad = ib->Concat({meta_grad_up, meta_grad_down}, -2);
  meta_grad = ib->Emit("MatrixExp", {meta_grad});
  return {ib->Slice(meta_grad, begins, sizes)};
});

REG_BPROP_BUILDER("Mv").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &vec = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  ShapeVector perm = {1, 0};

  // self: ger(dout, vec.conj())
  auto dx = ib->Outer(dout, vec);
  // vec: self.conj().t().mv(grad) -> dvec = ib->Mv(x^T, dout)
  auto dvec = ib->Mv(ib->Transpose(x, perm), dout);
  return {dx, dvec};
});

REG_BPROP_BUILDER("Complex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = x->need_compute_grad_out() ? ib->Real(dout) : ib->OutZeros(x);
  auto dy = y->need_compute_grad_out() ? ib->Imag(dout) : ib->OutZeros(y);
  return {dx, dy};
});

REG_BPROP_BUILDER("CholeskyInverse").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(i0);
  const auto &upper = ib->GetInput(i1);
  auto out = ib->GetInput(i2);
  auto dout = ib->GetInput(i3);
  ShapeVector input_perm = {1, 0};
  NodePtr dx;
  auto DealWithUpper = [&upper, &dx, &ib, &input_x](const NodePtr &common_term) {
    if (ib->Equal(upper, ib->Value<bool>(true))) {
      dx = ib->Neg(ib->MatMul(input_x, common_term, false, false));
    } else {
      dx = ib->Neg(ib->MatMul(common_term, input_x, false, false));
    }
  };
  if ((ib->GetDtypeId(dout)) == kNumberTypeFloat64) {
    input_x = ib->Cast(input_x, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
    auto common_term = ib->Add(dout, ib->Transpose(dout, input_perm));
    common_term = ib->Cast(common_term, kFloat32);
    common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
    DealWithUpper(common_term);
    dx = ib->Cast(dx, kFloat64);
    return {dx, ib->OutZeros(upper)};
  }
  auto common_term = ib->Add(dout, ib->Transpose(dout, input_perm));
  common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
  DealWithUpper(common_term);
  return {dx, ib->OutZeros(upper)};
});

REG_BPROP_BUILDER("CholeskySolve").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto upper = GetValue<bool>(ib->GetAttr("upper"));
  const auto &x1 = ib->GetInput(i0);
  auto x2 = ib->GetInput(i1);
  auto out = ib->GetInput(i2);
  auto dout = ib->GetInput(i3);
  auto flag = 0;
  auto shape_x1 = ib->GetShape(x1);
  auto len_x1 = shape_x1.size();
  if ((ib->GetDtypeId(dout)) == kNumberTypeFloat64) {
    flag = 1;
    x2 = ib->Cast(x2, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  ShapeVector input_perm = {1, 0};
  NodePtr dx2;
  auto DealWithUpper2D = [&upper, &dx2, &ib, &x2](const NodePtr &common_term) {
    if (upper) {
      dx2 = ib->Neg(ib->MatMul(x2, common_term, false, false));
    } else {
      dx2 = ib->Neg(ib->MatMul(common_term, x2, false, false));
    }
  };
  auto DealWithUpperND = [&upper, &dx2, &ib, &x2](const NodePtr &common_term) {
    if (upper) {
      dx2 = ib->Neg(ib->BatchMatMul(x2, common_term, false, false));
    } else {
      dx2 = ib->Neg(ib->BatchMatMul(common_term, x2, false, false));
    }
  };
  auto dx1 = ib->Emit("CholeskySolve", {dout, x2}, {{"upper", ib->GetAttr("upper")}});
  if (len_x1 == 2) {
    auto common_term = ib->MatMul(dx1, ib->Transpose(out, input_perm), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, input_perm)));
    DealWithUpper2D(common_term);
  } else {
    auto x2_dim_size = static_cast<int64_t>(ib->GetShape(x2).size());
    auto target_order = Range(x2_dim_size - 2);
    target_order.push_back(x2_dim_size - 1);
    target_order.push_back(x2_dim_size - 2);
    auto common_term = ib->BatchMatMul(dx1, ib->Transpose(out, target_order), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, target_order)));
    DealWithUpperND(common_term);
  }
  if (flag == 1) {
    dx1 = ib->Cast(dx1, kFloat64);
    dx2 = ib->Cast(dx2, kFloat64);
  }
  return {dx1, dx2};
});

REG_BPROP_BUILDER("NextAfter").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x1 = ib->GetInput(i0);
  auto x2 = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  auto dout_type = ib->GetDtype(dout);
  auto x1_type = ib->GetDtype(x1);
  if (x1_type->type_id() == kNumberTypeFloat64) {
    x1 = ib->Cast(x1, kFloat32);
  }
  if (ib->GetDtypeId(x2) == kNumberTypeFloat64) {
    x2 = ib->Cast(x2, kFloat32);
  }
  if (dout_type->type_id() == kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto s_x1 = ib->Shape(x1);
  auto partial_x1 = ib->Fill(1.0, s_x1, x1_type->type_id());
  auto s_x2 = ib->Shape(x2);
  auto partial_x2 = ib->ZerosLike(x2);
  auto dx1 = ib->Reshape(ib->Mul(partial_x1, dout), s_x1);
  auto dx2 = ib->Reshape(ib->Mul(partial_x2, dout), s_x2);
  return {ib->Cast(dx1, dout_type), ib->Cast(dx2, dout_type)};
});

REG_BPROP_BUILDER("LinalgVectorNorm").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto input_shape = ib->GetShape(input);
  const auto &ord = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &dtype = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  if (ord->BuildValue()->ContainsValueAny()) {
    MS_EXCEPTION(ValueError) << "For gradient of `LinalgVectorNorm`, `ord` must be constant!";
  }
  auto grad_input = VectorNormGrad(ib, input, ord, dim, keepdim, out, dout);
  return {grad_input, ib->OutZeros(ord), ib->OutZeros(dim), ib->OutZeros(keepdim), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("Norm").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto input_shape = ib->GetShape(input);
  const auto &p = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &dtype = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  if (p->BuildValue()->ContainsValueAny()) {
    MS_EXCEPTION(ValueError) << "For gradient of `Norm`, `p` must be constant!.";
  }
  auto grad_input = VectorNormGrad(ib, input, p, dim, keepdim, out, dout);
  return {grad_input, ib->OutZeros(p), ib->OutZeros(dim), ib->OutZeros(keepdim), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("LpNormV2").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto input_shape = ib->GetShape(input);
  const auto &p = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &epsilon = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto dim_type = dim->abstract()->BuildType();
  if (p->BuildValue()->ContainsValueAny()) {
    MS_EXCEPTION(ValueError) << "For gradient of `LpNormV2`, `p` must be constant!";
  }
  auto grad_input = VectorNormGrad(ib, input, p, dim, keepdim, out, dout);
  return {grad_input, ib->OutZeros(p), ib->OutZeros(dim), ib->OutZeros(keepdim), ib->OutZeros(epsilon)};
});

REG_BPROP_BUILDER("Lerp").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &start = ib->GetInput(i0);
  const auto &end = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dout_type = ib->GetDtype(dout);
  NodePtr dstart = nullptr;
  NodePtr dend = nullptr;
  NodePtr dweight = nullptr;
  if (start->need_compute_grad_out()) {
    NodePtr sub_w = ib->Sub(ib->Tensor(1.0, ib->GetDtype(weight)), weight);
    dstart = ib->Mul(dout, sub_w);
    dstart = BroadcastReduceToInputShape(ib, start, dout, dstart);
  } else {
    dstart = ib->OutZeros(start);
  }
  if (end->need_compute_grad_out()) {
    dend = ib->Mul(dout, weight);
    dend = BroadcastReduceToInputShape(ib, end, dout, dend);
  } else {
    dend = ib->OutZeros(end);
  }
  if (weight->need_compute_grad_out()) {
    dweight = ib->Mul(dout, ib->Sub(end, start));
    dweight = BroadcastReduceToInputShape(ib, weight, dout, dweight);
  } else {
    dweight = ib->OutZeros(weight);
  }
  return {dstart, dend, dweight};
});

REG_BPROP_BUILDER("LerpScalar").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &start = ib->GetInput(i0);
  const auto &end = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto val = GetScalarValue<float>(weight->BuildValue());
  NodePtr dstart = val.has_value() ? ib->Muls(dout, ib->Value<float>(1.0 - static_cast<double>(val.value())))
                                   : ib->Muls(dout, ib->ScalarSub(ib->Value<float>(1.0), weight));
  auto dend = ib->Muls(dout, weight);
  auto tmp = BinopGradCommon(ib, start, end, dstart, dend);
  dstart = tmp[0];
  dend = tmp[1];
  return {dstart, dend, ib->OutZeros(weight)};
});

REG_BPROP_BUILDER("TridiagonalMatMul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto LeftShift = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    (void)paddings.emplace_back(ShapeVector{0LL, 1LL});
    (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      (void)begin.emplace_back(0LL);
      (void)end.emplace_back(x_shape[i]);
      (void)strides.emplace_back(1LL);
    }
    begin[rank - 2] = 1LL;
    return ib->Emit("Pad",
                    {ib->StridedSlice(x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                                      ib->Value<ShapeVector>(strides))},
                    {{"paddings", MakeValue(paddings)}});
  };
  auto RightShift = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    (void)paddings.emplace_back(ShapeVector{1LL, 0LL});
    (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      (void)begin.emplace_back(0LL);
      (void)end.emplace_back(x_shape[i]);
      (void)strides.emplace_back(1LL);
    }
    end[rank - 2] = -1;
    return ib->Emit("Pad",
                    {ib->StridedSlice(x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                                      ib->Value<ShapeVector>(strides))},
                    {{"paddings", MakeValue(paddings)}});
  };
  auto MatrixTranspose = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    auto rank = x_shape.size();
    ShapeVector perm;
    if (rank > IntToSize(2)) {
      for (size_t i = 0; i < rank - 2; i++) {
        (void)perm.emplace_back(SizeToLong(i));
      }
    }
    (void)perm.emplace_back(rank - 1);
    (void)perm.emplace_back(rank - 2);
    return ib->Transpose(x, perm);
  };
  auto superdiag = ib->GetInput(i0);
  auto maindiag = ib->GetInput(i1);
  auto subdiag = ib->GetInput(i2);
  auto rhs = ib->GetInput(i3);
  auto dout = ib->GetInput(i5);
  auto superdiag_type = ib->GetDtype(superdiag);
  auto superdiag_conj = MatrixTranspose(ib, superdiag);
  auto maindiag_conj = MatrixTranspose(ib, maindiag);
  auto subdiag_conj = MatrixTranspose(ib, subdiag);
  auto rhs_conj = rhs;
  if ((*superdiag_type) == (*kComplex64) || (*superdiag_type) == (*kComplex128)) {
    superdiag_conj = ib->Conj(superdiag_conj);
    maindiag_conj = ib->Conj(maindiag_conj);
    subdiag_conj = ib->Conj(subdiag_conj);
    rhs_conj = ib->Conj(rhs);
  }
  auto superdiag_grad = ib->ReduceSum(LeftShift(ib, rhs_conj) * dout, ShapeVector{-1LL});
  auto maindiag_grad = ib->ReduceSum(rhs_conj * dout, ShapeVector{-1LL});
  auto subdiag_grad = ib->ReduceSum(RightShift(ib, rhs_conj) * dout, ShapeVector{-1LL});
  auto rhs_grad = RightShift(ib, superdiag_conj * dout) + maindiag_conj * dout + LeftShift(ib, subdiag_conj * dout);
  superdiag_grad = ib->ExpandDims(superdiag_grad, -2);
  maindiag_grad = ib->ExpandDims(maindiag_grad, -2);
  subdiag_grad = ib->ExpandDims(subdiag_grad, -2);
  return {superdiag_grad, maindiag_grad, subdiag_grad, rhs_grad};
});

REG_BPROP_BUILDER("AddV2").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = dout;
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Addcdiv").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) { return BpropAddcCommon(ib, "Addcdiv"); });

REG_BPROP_BUILDER("AddcdivExt").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &tensor1 = ib->GetInput(i1);
  const auto &tensor2 = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);

  auto shape = ib->GetShape(tensor2);
  auto dtype = ib->GetDtype(tensor2)->type_id();
  auto bc_dinput = dout;

  if (input->need_compute_grad_out()) {
    bc_dinput = BinopGradCommon(ib, out, input, dout, bc_dinput)[1];
  } else {
    bc_dinput = ib->OutZeros(input);
  }
  auto bc_dtensor1 =
    tensor1->need_compute_grad_out()
      ? BinopGradCommon(
          ib, out, tensor1, dout,
          ib->Mul(dout, ib->Div(ib->FillScalar(ib->Value(shape), value, ib->Value((int64_t)dtype)), tensor2)))[1]
      : ib->OutZeros(tensor1);
  auto bc_dtensor2 =
    tensor2->need_compute_grad_out()
      ? BinopGradCommon(ib, out, tensor2, dout,
                        ib->Neg(ib->Div(ib->Mul(dout, ib->Muls(tensor1, value)), ib->Mul(tensor2, tensor2))))[1]
      : ib->OutZeros(tensor2);
  return {bc_dinput, bc_dtensor1, bc_dtensor2, ib->OutZeros(value)};
});

REG_BPROP_BUILDER("Addcmul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) { return BpropAddcCommon(ib, "Addcmul"); });

REG_BPROP_BUILDER("AddcmulExt").SetUnusedInputs({}).SetBody(BODYFUNC(ib) {
  const auto &input_data = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &y = ib->GetInput(i2);
  const auto &value = ib->GetInput(i3);
  const auto &inner_out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto dinput_data = dout;
  auto dout_typeptr = ib->GetDtype(dout);
  if (input_data->need_compute_grad_out()) {
    dinput_data = BinopGradCommon(ib, inner_out, input_data, dout, dinput_data)[i1];
  } else {
    dinput_data = ib->OutZeros(input_data);
  }
  auto dx = x->need_compute_grad_out() ? BinopGradCommon(ib, inner_out, x, dout, ib->Mul(dout, ib->Muls(y, value)))[i1]
                                       : ib->OutZeros(x);
  auto dy = y->need_compute_grad_out() ? BinopGradCommon(ib, inner_out, y, dout, ib->Mul(dout, ib->Muls(x, value)))[i1]
                                       : ib->OutZeros(y);
  return {dinput_data, dx, dy, ib->OutZeros(value)};
});

REG_BPROP_BUILDER("LpNorm").SetBody(BODYFUNC(ib) {
  auto p = GetValue<int64_t>(ib->GetAttr("p"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto axis = GetIntList(ib->GetAttr("axis"));
  const auto &input_x = ib->GetInput(i0);
  auto out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto input_x_shape = ib->GetShape(input_x);
  if ((!keep_dims) && (!input_x_shape.empty())) {
    for (const auto &i : axis) {
      dout = ib->ExpandDims(dout, i);
      out = ib->ExpandDims(out, i);
    }
  }
  if (p == 0) {
    return {ib->OutZeros(input_x)};
  }
  if (p == 1) {
    return {ib->Mul(dout, (ib->Sign(input_x)))};
  }
  if (p == 2) {
    auto input_scaled = input_x;
    auto scale_v = ib->RealDiv(dout, out);
    return {ib->Mul(input_scaled, scale_v)};
  } else {
    auto input_x_abs = ib->Abs(input_x);
    auto input_scaled =
      ib->Mul(ib->PowTensorScalar(input_x_abs, ib->ValueByType(p - 2, ib->GetDtype(input_x_abs))), input_x);
    auto scale_v = ib->RealDiv(dout, ib->PowTensorScalar(out, ib->ValueByType(p - 1, ib->GetDtype(out))));
    auto equal_zero = ib->Equal(input_scaled, ib->Tensor(0, ib->GetDtype(input_scaled)));
    return {ib->Select(equal_zero, ib->Fill(0.0, ib->Shape(input_scaled), ib->GetDtype(input_scaled)->type_id()),
                       ib->Mul(input_scaled, scale_v))};
  }
});

REG_BPROP_BUILDER("Renorm").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto p = static_cast<int64_t>(GetValue<float>(ib->GetAttr("p")));
  float ext = 1e-07;
  auto dim = GetIntList(ib->GetAttr("dim"))[0];
  auto max_norm = GetValue<float>(ib->GetAttr("maxnorm"));
  const auto &input_x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto shape = ib->GetShape(input_x);
  int64_t new_dim = dim >= 0 ? dim : (SizeToLong(shape.size()) + dim);
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < SizeToLong(shape.size()); i++) {
    if (i != new_dim) {
      dims.push_back(i);
    }
  }
  auto norm = ib->Emit("LpNorm", {input_x},
                       {{"keep_dims", MakeValue(true)},
                        {"axis", MakeValue(dims)},
                        {"p", MakeValue<int64_t>(p)},
                        {"epsilon", MakeValue<float>(1e-12)}});
  norm = ib->BroadcastTo(norm, input_x);
  auto grad_out = ib->Mul(input_x, dout);
  grad_out = ib->ReduceSum(grad_out, dims, true);
  NodePtr norm_bp = nullptr;
  if (p == 1) {
    auto sig = ib->Sign(input_x);
    norm_bp = ib->Mul(sig, grad_out);
  } else if (p == 2) {
    auto m = ib->Mul(input_x, (ib->RealDiv(grad_out, norm)));
    norm_bp =
      ib->MaskedFill(m, ib->Equal(norm, (ib->Tensor(0.0, ib->GetDtype(norm)))), ib->Tensor(0.0, ib->GetDtype(m)));
  } else {
    auto abs_ = ib->Abs(input_x);
    auto input_scaled = ib->Mul(input_x, ib->PowTensorScalar(abs_, ib->Value(p - 2)));
    auto pow_ = ib->PowTensorScalar(norm, ib->Value(p - 1));
    auto scale_v = ib->RealDiv(grad_out, pow_);
    scale_v = ib->MaskedFill(scale_v, ib->Equal(norm, (ib->Tensor(0.0, ib->GetDtype(norm)))),
                             ib->Tensor(0.0, ib->GetDtype(scale_v)));
    norm_bp = ib->Mul(input_scaled, scale_v);
  }

  auto v = ib->AddScalar(norm, ib->ValueByType(ext, ib->GetDtype(norm)), ib->Value(1));
  auto inv_norm = ib->Reciprocal(v);
  auto grad_norm = ib->Mul(ib->Muls(inv_norm, ib->ValueByType(max_norm, ib->GetDtype(inv_norm))),
                           ib->Sub(dout, (ib->Mul(inv_norm, norm_bp))));
  auto q = ib->Greater(norm, ib->Tensor(max_norm, ib->GetDtype(norm)));
  return {ib->Select(q, grad_norm, dout)};
});

DEF_PURE_SHAPE_CALC(g_reduce_std)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray { return ReduceStdShapeFunc(inputs.at(0), inputs.at(1)); })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> ShapeVector {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : SizeToLong(shape_x.size());
    return {rank, 1, 1};
  });

REG_BPROP_BUILDER("ReduceStd").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &unbiased = ib->GetInput(i2);
  const auto &keep_dims = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto std_d = ib->TupleGetItem(dout, 0);
  auto std = ib->TupleGetItem(out, 0);
  auto mean_d = ib->TupleGetItem(dout, 1);
  auto mean = ib->TupleGetItem(out, 1);
  auto res = ib->ShapeCalc(g_reduce_std, {x, axis}, {1});
  res[1] = ib->SequenceToTensor(res[1]);
  res[2] = ib->SequenceToTensor(res[2]);

  auto keep_dims_value = keep_dims->BuildValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (keep_dims_opt.has_value()) {
    if (!keep_dims_opt.value() && !ib->GetShape(x).empty()) {
      std_d = ib->Reshape(std_d, res[0]);
      std = ib->Reshape(std, res[0]);
      mean_d = ib->Reshape(mean_d, res[0]);
      mean = ib->Reshape(mean, res[0]);
    }
  } else {
    auto true_branch = [&res, &std_d, &std, &mean_d, &mean](Emitter *e) -> NodePtrList {
      auto std_d_r = e->Reshape(std_d, res[0]);
      auto std_r = e->Reshape(std, res[0]);
      auto mean_d_r = e->Reshape(mean_d, res[0]);
      auto mean_r = e->Reshape(mean, res[0]);
      return {std_d_r, std_r, mean_d_r, mean_r};
    };
    auto false_branch = [&std_d, &std, &mean_d, &mean](const Emitter *e) -> NodePtrList {
      return {std_d, std, mean_d, mean};
    };
    auto keep_dims_t = ib->ScalarToTensor(ib->Equal(keep_dims, ib->Value<bool>(false)), kBool);
    auto cond = ib->LogicalAnd(keep_dims_t, ib->Tensor(!(ib->GetShape(x).empty()), kBool));
    auto cond_block = ib->Conditional(cond, true_branch, false_branch);
    std_d = ib->TupleGetItem(cond_block, 0);
    std = ib->TupleGetItem(cond_block, 1);
    mean_d = ib->TupleGetItem(cond_block, 2);
    mean = ib->TupleGetItem(cond_block, 3);
  }

  auto dx = ib->Sub(x, mean);
  dx = ib->Mul(dx, std_d);
  auto dx_type = ib->GetDtype(dx);
  dx = ib->Cast(ib->Div(dx, std), dx_type);

  auto unbiased_value = unbiased->BuildValue();
  auto unbiased_opt = GetScalarValue<bool>(unbiased_value);
  if (unbiased_opt.has_value()) {
    if (unbiased_opt.value()) {
      dx = ib->Cast(ib->Div(dx, ib->Cast(res[1], ib->GetDtype(dx))), dx_type);
    } else {
      dx = ib->Cast(ib->Div(dx, ib->Cast(res[2], ib->GetDtype(dx))), dx_type);
    }
  } else {
    auto unbiased_true_branch = [&dx, &res](Emitter *e) -> NodePtrList {
      return {e->Cast(e->Div(dx, e->Cast(res[1], dx->dtype())), dx->dtype())};
    };
    auto unbiased_false_branch = [&dx, &res](Emitter *e) -> NodePtrList {
      return {e->Cast(e->Div(dx, e->Cast(res[2], dx->dtype())), dx->dtype())};
    };
    auto unbiased_cond = ib->Equal(unbiased, ib->Value<bool>(true));
    dx = ib->Conditional(unbiased_cond, unbiased_true_branch, unbiased_false_branch);
  }
  auto temp = ib->Cast(ib->Div(mean_d, ib->Cast(res[2], ib->GetDtype(mean_d))), ib->GetDtype(mean_d));
  dx = ib->Add(dx, temp);
  return {dx, ib->OutZeros(axis), ib->OutZeros(unbiased), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("CumulativeLogsumexp").SetBody(BODYFUNC(ib) {
  // this dsl has pression error
  const auto &x = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  bool reverse = GetValue<bool>(ib->GetAttr("reverse"));
  NodePtr dtype_min = nullptr;
  if ((ib->GetDtype(x))->type_id() == TypeId::kNumberTypeFloat16) {
    dtype_min = ib->Fill(-65500e+0, ib->Shape(dout), TypeId::kNumberTypeFloat16);
  } else {
    if ((ib->GetDtype(x))->type_id() == TypeId::kNumberTypeFloat32) {
      dtype_min = ib->Fill(-3.4028235e+38, ib->Shape(dout), TypeId::kNumberTypeFloat32);
    } else {
      dtype_min = ib->Fill(-1.7976931348623157e+308, ib->Shape(dout), TypeId::kNumberTypeFloat64);
    }
  }
  auto log_grad_positive = ib->Select(ib->Greater(dout, ib->Tensor(0, ib->GetDtype(dout))), ib->Log(dout), dtype_min);
  auto log_grad_negative =
    ib->Select(ib->Less(dout, ib->Tensor(0, ib->GetDtype(dout))), ib->Log(ib->Neg(dout)), dtype_min);
  auto output_pos =
    ib->Exp(ib->Add((ib->Emit("CumulativeLogsumexp", {ib->Sub(log_grad_positive, out), axis},
                              {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}})),
                    x));
  auto output_neg =
    ib->Exp(ib->Add((ib->Emit("CumulativeLogsumexp", {ib->Sub(log_grad_negative, out), axis},
                              {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}})),
                    x));
  return {ib->Sub(output_pos, output_neg), ib->OutZeros(x)};
});

REG_BPROP_BUILDER("NPUAllocFloatStatus").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) { return {}; });

REG_BPROP_BUILDER("NPUGetFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NPUClearFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Igamma").SetUnusedInputs({i2}).SetBody(IgammaBpropExpander);

REG_BPROP_BUILDER("Igammac").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto r_dx = IgammaBpropExpander(ib);
  r_dx[0] = ib->Neg(r_dx[0]);
  r_dx[1] = ib->Neg(r_dx[1]);
  return r_dx;
});

REG_BPROP_BUILDER("Einsum").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("EinsumGrad", {x, dout}, {{"equation", ib->GetAttr("equation")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchMatMul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &trans_a = ib->GetInput(i2);
  const auto &trans_b = ib->GetInput(i3);
  auto ta_opt = mindspore::GetScalarValue<bool>(trans_a->BuildValue());
  auto tb_opt = mindspore::GetScalarValue<bool>(trans_b->BuildValue());

  if (!ta_opt.has_value() || !tb_opt.has_value()) {
    MS_LOG_EXCEPTION << "For BatchMatMul, got invalid 'transpose_a' or 'transpose_b'.";
  }
  auto ta = ta_opt.value();
  auto tb = tb_opt.value();
  const auto &dout = ib->GetInput(i5);

  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    if (ta) {
      dx = ib->BatchMatMul(w, dout, tb, true);
    } else {
      dx = ib->BatchMatMul(dout, w, false, !tb);
    }
  }
  NodePtr dw = nullptr;
  if (w->need_compute_grad_out()) {
    if (tb) {
      dw = ib->BatchMatMul(dout, x, true, ta);
    } else {
      dw = ib->BatchMatMul(x, dout, !ta, false);
    }
  }

  std::vector<NodePtr> ret = BinopGradCommon(ib, x, w, dx, dw, 2);
  ret.emplace_back(ib->OutZeros(trans_a));
  ret.emplace_back(ib->OutZeros(trans_b));
  return ret;
});

REG_BPROP_BUILDER("BatchMatMulExt").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx;
  NodePtr dw;
  if (x->need_compute_grad_out()) {
    ShapeVector perm_w = {0, 2, 1};
    auto w_t = ib->Transpose(w, perm_w);
    dx = ib->BatchMatMulExt(dout, w_t);
  } else {
    dx = ib->OutZeros(x);
  }
  if (w->need_compute_grad_out()) {
    ShapeVector perm_x = {0, 2, 1};
    auto x_t = ib->Transpose(x, perm_x);
    dw = ib->BatchMatMulExt(x_t, dout);
  } else {
    dw = ib->OutZeros(w);
  }
  return {dx, dw};
});

REG_BPROP_BUILDER("Eps").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("EuclideanNorm").SetBody(BODYFUNC(ib) {
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  const auto &x = ib->GetInput(i0);
  const auto &axes = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto scale_v = ib->RealDiv(dout, out);
  if ((!keep_dims) && (ib->GetShape(x).size() > 0)) {
    scale_v = ib->ExpandDims(scale_v, axes);
  }
  return {ib->Mul(x, scale_v), ib->OutZeros(axes)};
});

REG_BPROP_BUILDER("MatrixTriangularSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto adjoint_a = GetValue<bool>(ib->GetAttr("adjoint"));
  auto lower_a = GetValue<bool>(ib->GetAttr("lower"));
  const auto &matrix = ib->GetInput(i0);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto grad_rhs = ib->Emit("MatrixTriangularSolve", {matrix, dout},
                           {{"lower", MakeValue(lower_a)}, {"adjoint", MakeValue(!adjoint_a)}});
  NodePtr grad_rhs_temp;
  NodePtr out_temp;
  if (((ib->GetDtype(matrix)) == kComplex64) || ((ib->GetDtype(matrix)) == kComplex128)) {
    grad_rhs_temp = Adjoint(ib, grad_rhs);
    out_temp = Adjoint(ib, out);
  } else {
    grad_rhs_temp = MatrixTranspose(ib, grad_rhs);
    out_temp = MatrixTranspose(ib, out);
  }
  NodePtr grad_matrix;
  auto m_rank = ib->GetShape(matrix).size();
  auto NegMatMul = [&ib, &grad_matrix, &m_rank](const NodePtr &a, const NodePtr &b) {
    if (m_rank == 2) {
      grad_matrix = ib->MatMul(a, b, false, false);
    } else {
      grad_matrix = ib->BatchMatMul(a, b, false, false);
    }
    grad_matrix = ib->Neg(grad_matrix);
  };
  if (adjoint_a) {
    NegMatMul(out, grad_rhs_temp);
  } else {
    NegMatMul(grad_rhs, out_temp);
  }
  auto BandPart = [&ib](const NodePtr &matrix, int lower, int upper) -> NodePtr {
    if (((ib->GetDtype(matrix)) == kComplex64) || ((ib->GetDtype(matrix)) == kComplex128)) {
      auto grad_matrix_real = ib->Emit("MatrixBandPart", {ib->Real(matrix), ib->Value(lower), ib->Value(upper)});
      auto grad_matrix_imag = ib->Emit("MatrixBandPart", {ib->Imag(matrix), ib->Value(lower), ib->Value(upper)});
      return ib->Emit("Complex", {grad_matrix_real, grad_matrix_imag});
    } else {
      return ib->Emit("MatrixBandPart", {matrix, ib->Value(lower), ib->Value(upper)});
    }
  };
  if (lower_a) {
    grad_matrix = BandPart(grad_matrix, -1, 0);
  } else {
    grad_matrix = BandPart(grad_matrix, 0, -1);
  }
  return {grad_matrix, grad_rhs};
});

REG_BPROP_BUILDER("TriangularSolve").FreeUselessValues_O({i1}).SetBody(BODYFUNC(ib) {
  const auto &b = ib->GetInput(i0);
  const auto &A = ib->GetInput(i1);
  const auto &upper = ib->GetInput(i2);
  const auto &transpose = ib->GetInput(i3);
  const auto &unitriangular = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  NodePtr grad_b = nullptr;
  NodePtr grad_A = nullptr;
  auto grad_x = ib->TupleGetItem(dout, i0);
  auto grad_m = ib->TupleGetItem(dout, i1);
  auto grad_x_defined = !grad_x->BuildValue()->isa<None>();
  auto grad_m_defined = !grad_m->BuildValue()->isa<None>();
  auto x = ib->TupleGetItem(out, i0);

  if (!grad_x_defined && !grad_m_defined) {
    return {grad_b, grad_A, ib->OutZeros(upper), ib->OutZeros(transpose), ib->OutZeros(unitriangular)};
  }

  if (grad_x_defined) {
    auto transpose_opt = mindspore::GetScalarValue<bool>(transpose->BuildValue());
    if (!transpose_opt.has_value()) {
      auto true_branch = [&](Emitter *e) -> NodePtrList {
        return {
          e->TupleGetItem(e->TriangularSolve(grad_x, e->Conj(A), upper, e->Value<bool>(false), unitriangular), i0)};
      };
      auto false_branch = [&](Emitter *e) -> NodePtrList {
        return {
          e->TupleGetItem(e->TriangularSolve(grad_x, e->Conj(A), upper, e->Value<bool>(true), unitriangular), i0)};
      };
      auto transpose_opt_true = ib->Equal(transpose, ib->Value<bool>(true));
      grad_b = ib->Conditional(transpose_opt_true, true_branch, false_branch);
    } else {
      grad_b = ib->TupleGetItem(
        ib->TriangularSolve(grad_x, ib->Conj(A), upper, ib->Value<bool>(!transpose_opt.value()), unitriangular), i0);
    }
    if (A->need_compute_grad_out()) {
      if (!transpose_opt.has_value()) {
        auto true_branch = [&](Emitter *e) -> NodePtrList {
          return {e->MatMulExt(e->Conj(x), e->TransposeExtView(grad_b, e->Value<int64_t>(-1), e->Value<int64_t>(-2)))};
        };
        auto false_branch = [&](Emitter *e) -> NodePtrList {
          return {e->MatMulExt(grad_b, e->TransposeExtView(x, e->Value<int64_t>(-1), e->Value<int64_t>(-2)))};
        };
        auto transpose_opt_true = ib->Equal(transpose, ib->Value<bool>(true));
        grad_A = ib->Conditional(transpose_opt_true, true_branch, false_branch);
      } else {
        grad_A =
          transpose_opt.value()
            ? ib->MatMulExt(ib->Conj(x), ib->TransposeExtView(grad_b, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)))
            : ib->MatMulExt(grad_b, ib->TransposeExtView(x, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)));
      }
      grad_A = ib->Neg(grad_A);
      auto unitriangular_opt = mindspore::GetScalarValue<bool>(unitriangular->BuildValue());
      NodePtr unitriangular_int, unitriangular_int_neg;
      if (!unitriangular_opt.has_value()) {
        auto true_branch = [&](Emitter *e) -> NodePtrList { return {e->Value<int64_t>(1)}; };
        auto false_branch = [&](Emitter *e) -> NodePtrList { return {e->Value<int64_t>(0)}; };
        auto unitriangular_opt_true = ib->Equal(unitriangular, ib->Value<bool>(true));
        unitriangular_int = ib->Conditional(unitriangular_opt_true, true_branch, false_branch);
        unitriangular_int_neg = ib->ScalarNeg(unitriangular_int);
      } else {
        unitriangular_int = unitriangular_opt.value() ? ib->Value<int64_t>(1) : ib->Value<int64_t>(0);
        unitriangular_int_neg = unitriangular_opt.value() ? ib->Value<int64_t>(-1) : ib->Value<int64_t>(0);
      }

      auto upper_opt = mindspore::GetScalarValue<bool>(upper->BuildValue());
      if (!upper_opt.has_value()) {
        auto true_branch = [&](Emitter *e) -> NodePtrList { return {e->Triu(grad_A, unitriangular_int)}; };
        auto false_branch = [&](Emitter *e) -> NodePtrList { return {e->TrilExt(grad_A, unitriangular_int_neg)}; };
        auto upper_opt_true = ib->Equal(upper, ib->Value<bool>(true));
        grad_A = ib->Conditional(upper_opt_true, true_branch, false_branch);
      } else {
        grad_A = upper_opt.value() ? ib->Triu(grad_A, unitriangular_int) : ib->TrilExt(grad_A, unitriangular_int_neg);
      }
    }
  }
  if (grad_b == nullptr) {
    grad_b = ib->OutZeros(b);
  }
  if (grad_A == nullptr) {
    grad_A = ib->OutZeros(A);
  }
  if (A->need_compute_grad_out() && grad_m_defined) {
    grad_A = ib->Add(grad_A, grad_m);
  }

  auto tmp = BinopGradCommon(ib, b, A, grad_b, grad_A, 2);
  grad_b = tmp[0];
  grad_A = tmp[1];
  return {grad_b, grad_A, ib->OutZeros(upper), ib->OutZeros(transpose), ib->OutZeros(unitriangular)};
});

REG_BPROP_BUILDER("NanToNum").FreeUselessValues_IO({i1, i2, i3}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &nan = ib->GetInput(i1);
  const auto &posinf = ib->GetInput(i2);
  const auto &neginf = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->Mul(dout, (ib->IsFinite(x)));
  return {dx, ib->OutZeros(nan), ib->OutZeros(posinf), ib->OutZeros(neginf)};
});

REG_BPROP_BUILDER("Fmin").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) { return FminFmaxGrad(ib, true); });

REG_BPROP_BUILDER("Fmax").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) { return FminFmaxGrad(ib, false); });

REG_BPROP_BUILDER("Angle").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto re = ib->Real(x);
  auto im = ib->Imag(x);
  re = ib->Complex(im, re);
  auto z = ib->Reciprocal(re);
  auto zero = ib->ZerosLike(dout);
  auto complex_dout = ib->Complex(dout, zero);
  return {ib->Neg(ib->Mul(complex_dout, z))};
});

REG_BPROP_BUILDER("Lgamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, ib->Emit(kDigammaOpName, {x}));
  return {dx};
});

REG_BPROP_BUILDER("Digamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto a = ib->Tensor(1, kInt64);
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
  return {dx};
});

REG_BPROP_BUILDER("Polygamma").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto a = ib->GetInput(i0);
  auto x = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto one = ib->Value(1);
  a = ib->AddScalar(a, one, ib->Value(1));
  NodePtr dx;
  if (ib->GetDtypeId(x) == kNumberTypeFloat16) {
    x = ib->Cast(x, kNumberTypeFloat64);
    dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
    dx = ib->Cast(dx, kNumberTypeFloat16);
  } else {
    dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
  }
  return {ib->OutZeros(a), dx};
});

REG_BPROP_BUILDER("Cholesky").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &upper_input = ib->GetInput(i1);
  auto upper_input_value = upper_input->BuildValue();
  if (upper_input_value->ContainsValueAny()) {
    MS_EXCEPTION(ValueError) << "Input `upper` does not support variable in GRAPH_MODE currently.";
  }
  auto upper = GetValue<bool>(upper_input_value);
  auto out = ib->GetInput(i2);
  auto dout = ib->GetInput(i3);
  if (upper) {
    out = MatrixTranspose(ib, out);
    dout = MatrixTranspose(ib, dout);
  }
  auto dx = ib->Emit("CholeskyGrad", {out, dout});
  return {dx, ib->OutZeros(upper_input)};
});

REG_BPROP_BUILDER("InplaceIndexAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  return {dout, ib->OutZeros(indices), ib->Gather(dout, indices, ib->EmitValue(ib->GetAttr("axis")))};
});

REG_BPROP_BUILDER("Addmm").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &mat1 = ib->GetInput(i1);
  const auto &mat2 = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &alpha = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  NodePtr input_grad{nullptr};
  auto input_type = ib->GetDtype(input);
  if (input->need_compute_grad_out()) {
    input_grad = MaybeMultiply(ib, input_type, dout, beta, "beta");
    input_grad = BroadcastReduceToInputShape(ib, input, dout, input_grad);
  } else {
    input_grad = ib->OutZeros(input);
  }
  auto mat1_grad =
    mat1->need_compute_grad_out() ? MmMat1Backward(ib, input_type, dout, mat1, mat2, alpha) : ib->OutZeros(mat1);
  NodePtr mat2_grad =
    mat2->need_compute_grad_out() ? MmMat2Backward(ib, input_type, dout, mat1, mat2, alpha) : ib->OutZeros(mat2);
  return {input_grad, mat1_grad, mat2_grad, ib->OutZeros(beta), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("IndexAddExt").SetUnusedInputs({i0, i5}).SetBody(BODYFUNC(ib) { return IndexAddGrad(ib); });

REG_BPROP_BUILDER("InplaceIndexAddExt").SetUnusedInputs({i0, i5}).SetBody(BODYFUNC(ib) { return IndexAddGrad(ib); });

REG_BPROP_BUILDER("InplaceAddmm").SetUnusedInputs({i0, i3, i5}).SetBody(BODYFUNC(ib) {
  const auto &mat1 = ib->GetInput(i1);
  const auto &mat2 = ib->GetInput(i2);
  auto mat1_t_conj = ib->Conj(ib->Transpose(mat1, {1, 0}));
  auto mat2_t_conj = ib->Conj(ib->Transpose(mat2, {1, 0}));
  auto aplha_conj = ib->Conj(ib->GetInput(i4));
  auto aplha_tensor = ib->ScalarToTensor(aplha_conj, ib->GetDtype(mat1));
  const auto &dout = ib->GetInput(i6);
  auto dmat1 = ib->Mul(ib->MatMul(dout, mat2_t_conj), aplha_tensor);
  auto dmat2 = ib->Mul(ib->MatMul(mat1_t_conj, dout), aplha_tensor);
  return {ib->OutZeros(dout), dmat1, dmat2, ib->OutZeros(aplha_conj), ib->OutZeros(aplha_conj)};
});

REG_BPROP_BUILDER("Zeta").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &q = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dq =
    ib->Mul((ib->Mul((ib->Neg(x)),
                     (ib->Emit("Zeta", {ib->AddScalar(x, ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1)), q})))),
            dout);
  return {ib->OutZeros(x), dq};
});

REG_BPROP_BUILDER("Addmv").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &mat = ib->GetInput(i1);
  const auto &vec = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &alpha = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  NodePtr input_grad{nullptr};
  auto input_type = ib->GetDtype(input);
  if (input->need_compute_grad_out()) {
    input_grad = MaybeMultiply(ib, input_type, dout, beta, "beta");
    input_grad = BroadcastReduceToInputShape(ib, input, dout, input_grad);
  } else {
    input_grad = ib->OutZeros(input);
  }
  auto dmat = mat->need_compute_grad_out() ? MaybeMultiply(ib, input_type, ib->Outer(dout, vec), alpha, "alpha")
                                           : ib->OutZeros(mat);
  auto dvec = vec->need_compute_grad_out()
                ? MaybeMultiply(ib, input_type, ib->Mv(ib->Transpose(mat, {1, 0}), dout), alpha, "alpha")
                : ib->OutZeros(vec);

  return {input_grad, dmat, dvec, ib->OutZeros(beta), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Addbmm").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &batch1 = ib->GetInput(i1);
  const auto &batch2 = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &alpha = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  NodePtr input_grad{nullptr};
  auto input_type = ib->GetDtype(input);
  if (input->need_compute_grad_out()) {
    input_grad = MaybeMultiply(ib, input_type, dout, beta, "beta");
    input_grad = BroadcastReduceToInputShape(ib, input, dout, input_grad);
  } else {
    input_grad = ib->OutZeros(input);
  }
  auto grad_unsqueeze = ib->ExpandDims(dout, 0);
  auto expand_shape = ib->ShapeCalc(g_addbmm_shapecalc, {batch1, batch2})[i0];
  auto grad_unsqueeze_expand = ib->Emit("BroadcastTo", {grad_unsqueeze, expand_shape});
  auto batch1_grad =
    batch1->need_compute_grad_out()
      ? MaybeMultiply(ib, input_type, ib->BatchMatMul(grad_unsqueeze_expand, batch2, false, true), alpha, "alpha")
      : ib->OutZeros(batch1);
  auto batch2_grad =
    batch2->need_compute_grad_out()
      ? MaybeMultiply(ib, input_type, ib->BatchMatMul(batch1, grad_unsqueeze_expand, true, false), alpha, "alpha")
      : ib->OutZeros(batch2);
  // inverse broadcast should be taken into consideration when addbmm is turned to stable
  return {input_grad, batch1_grad, batch2_grad, ib->OutZeros(beta), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Baddbmm").FreeUselessValues(FreeTensorsOfBaddbmm).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &batch1 = ib->GetInput(i1);
  const auto &batch2 = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &alpha = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  NodePtr input_grad{nullptr};
  auto input_type = ib->GetDtype(input);
  if (input->need_compute_grad_out()) {
    input_grad = MaybeMultiply(ib, input_type, dout, beta, "beta");
    input_grad = BroadcastReduceToInputShape(ib, input, dout, input_grad);
  } else {
    input_grad = ib->OutZeros(input);
  }
  auto batch1_grad = batch1->need_compute_grad_out()
                       ? MaybeMultiply(ib, input_type, ib->BatchMatMul(dout, batch2, false, true), alpha, "alpha")
                       : ib->OutZeros(batch1);
  auto batch2_grad = batch2->need_compute_grad_out()
                       ? MaybeMultiply(ib, input_type, ib->BatchMatMul(batch1, dout, true, false), alpha, "alpha")
                       : ib->OutZeros(batch2);
  return {input_grad, batch1_grad, batch2_grad, ib->OutZeros(beta), ib->OutZeros(alpha)};
});

NodePtrList DiagonalGrad(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto offset_node = ib->GetInput(i1);
  auto dim1_node = ib->GetInput(i2);
  auto dim2_node = ib->GetInput(i3);
  auto offset = GetValue<int64_t>(offset_node->BuildValue());
  auto dim1 = GetValue<int64_t>(dim1_node->BuildValue());
  auto dim2 = GetValue<int64_t>(dim2_node->BuildValue());
  auto out = ib->GetInput(i4);
  auto dout = ib->GetInput(i5);
  auto x_shape = ib->GetShape(x);
  if (IsDynamicRank(x_shape)) {
    MS_LOG_EXCEPTION << "Diagonal doesn't support dynamic rank now, because it need rank of x to do transpose";
  }
  auto x_dtype = ib->GetDtype(x);
  auto x_dim = ib->GetRank(x);
  if (dim1 < 0) {
    dim1 += SizeToLong(x_dim);
  }
  if (dim2 < 0) {
    dim2 += SizeToLong(x_dim);
  }
  auto true_case = [offset, dim1, dim2, &x, &out, &dout, &x_shape, &x_dtype, &offset_node, &dim1_node, &dim2_node,
                    x_dim](Emitter *ib) -> NodePtrList {
    auto dx_trans_shape = ib->ShapeCalc(std::make_shared<DiagonalShapeCalc>(dim1, dim2), {x, out})[0];
    auto grad_diagonal = GradDiagonal(ib, dout, dx_trans_shape, {offset, dim1, dim2, x_dim}, x_dtype);
    return {grad_diagonal, ib->ZerosLike(offset_node), ib->ZerosLike(dim1_node), ib->ZerosLike(dim2_node)};
  };
  auto false_case = [&x, &x_dtype, &offset_node, &dim1_node, &dim2_node](Emitter *ib) -> NodePtrList {
    return {ib->ZerosLike(x), ib->ZerosLike(offset_node), ib->ZerosLike(dim1_node), ib->ZerosLike(dim2_node)};
  };
  if (IsDynamic(ib->GetShape(out))) {
    auto out_size = ib->Emit("Size", {out});
    auto cond = ib->Emit("scalar_eq", {out_size, ib->Value<int64_t>(0)});
    auto dx = ib->Conditional(cond, false_case, true_case);
    return {dx, ib->OutZeros(offset_node), ib->OutZeros(dim1_node), ib->OutZeros(dim2_node)};
  }
  if (ib->GetSize(out) > 0) {
    return true_case(ib);
  } else {
    return false_case(ib);
  }
}

REG_BPROP_BUILDER("Diagonal").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) { return DiagonalGrad(ib); });
REG_BPROP_BUILDER("DiagonalView").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) { return DiagonalGrad(ib); });

REG_BPROP_BUILDER("DiagExt").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &diagonal = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto x_shape = ib->GetShape(x);
  auto ndimension = x_shape.size();
  MS_ASSERT(ndimension == 1 || ndimension == 2);
  if (ndimension == 1 || x_shape[0] == x_shape[1]) {
    return {ib->DiagExt(dout, diagonal), ib->OutZeros(diagonal)};
  }
  auto x_dim = ib->GetRank(x);
  auto x_dtype = ib->GetDtype(x);
  auto true_case = [&x, &out, &dout, &x_shape, &x_dtype, &diagonal, x_dim](Emitter *ib) -> NodePtrList {
    auto dx_trans_shape = ib->ShapeCalc(std::make_shared<DiagonalShapeCalc>(0, 1), {x, out})[0];
    auto grad_diagonal = GradDiagonalScalarToTensor(ib, dout, dx_trans_shape, diagonal, {0, 1, x_dim}, x_dtype);
    return {grad_diagonal, ib->ZerosLike(diagonal)};
  };
  auto false_case = [&x, &x_dtype, &diagonal](Emitter *ib) -> NodePtrList {
    return {ib->ZerosLike(x), ib->ZerosLike(diagonal)};
  };
  if (IsDynamic(ib->GetShape(x))) {
    auto x_size = ib->Emit("Size", {x});
    auto cond = ib->Emit("scalar_eq", {x_size, ib->Value<int64_t>(0)});
    auto dx = ib->Conditional(cond, false_case, true_case);
    return {dx, ib->OutZeros(diagonal)};
  }
  if (ib->GetSize(x) > 0) {
    return true_case(ib);
  } else {
    return false_case(ib);
  }
});

REG_BPROP_BUILDER("Polar").FreeUselessValues_I({}).SetBody(BODYFUNC(ib) {
  const auto &input_abs = ib->GetInput(i0);
  const auto &input_angle = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto grad_conj = ib->Emit("Conj", {dout});
  NodePtr grad_abs, grad_angle;
  if (input_abs->need_compute_grad_out()) {
    auto broadcast_axes = ib->BroadcastGradientArgs(dout, input_abs);
    MS_EXCEPTION_IF_CHECK_FAIL(!broadcast_axes.empty(), "BroadcastGradientArgs out should not be empty!");
    auto reduction_axes = broadcast_axes[i1];
    NodePtr grad_abs_full = ib->Mul(grad_conj, ib->Sign(out));
    NodePtr reduced_grad_abs = ib->ReduceSum(grad_abs_full, reduction_axes, true, true);
    auto abs_shape_node = ib->Shape(input_abs);
    auto dx = ib->Reshape(reduced_grad_abs, abs_shape_node);
    grad_abs = ib->Real(dx);
  } else {
    grad_abs = ib->OutZeros(input_abs);
  }

  if (input_angle->need_compute_grad_out()) {
    auto i = ib->Complex(ib->Tensor(0.0, ib->GetDtype(input_angle)), ib->Tensor(1.0, ib->GetDtype(input_angle)));
    auto result_mul_1_j = ib->Mul(out, i);
    auto broadcast_axes = ib->BroadcastGradientArgs(dout, input_angle);
    MS_EXCEPTION_IF_CHECK_FAIL(!broadcast_axes.empty(), "BroadcastGradientArgs out should not be empty!");
    auto reduction_axes = broadcast_axes[i1];
    NodePtr grad_angle_full = ib->Mul(grad_conj, result_mul_1_j);
    NodePtr reduced_grad_angle = ib->ReduceSum(grad_angle_full, reduction_axes, true, true);
    auto angle_shape_node = ib->Shape(input_angle);
    auto dx = ib->Reshape(reduced_grad_angle, angle_shape_node);
    grad_angle = ib->Real(dx);
  } else {
    grad_angle = ib->OutZeros(input_angle);
  }
  return {grad_abs, grad_angle};
});

REG_BPROP_BUILDER("TridiagonalSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &diagonals = ib->GetInput(i0);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  constexpr int64_t kLast2 = -2;
  constexpr int64_t k2 = 2;
  auto diag1 = ib->StridedSlice(diagonals, {{kLast2, {1}}});
  auto diag_shape = ib->GetShape(diagonals);
  ShapeVector zeros1_shape(diag_shape.begin(), diag_shape.end() - i2);
  zeros1_shape.push_back(1);
  auto zeros1 = ib->Zeros(ib->Value<ShapeVector>(zeros1_shape), ib->Value<int64_t>(ib->GetDtypeId(diagonals)));
  auto superdiag1 = ib->Concat({ib->StridedSlice(diagonals, {{kLast2, {k2}}, {-1, {1, LLONG_MAX}}}), zeros1}, -1);
  auto subdiag1 = ib->Concat({zeros1, ib->StridedSlice(diagonals, {{kLast2, {0}}, {-1, {0, -1}}})}, -1);
  auto diags_transposed = ib->Stack({superdiag1, diag1, subdiag1}, kLast2);
  auto grad_rhs = ib->Emit("TridiagonalSolve", {diags_transposed, dout}, {{"partial_pivoting", MakeValue<bool>(true)}});
  NodePtr grad_diags;
  if (diagonals->need_compute_grad_out()) {
    auto diag2 = ib->ReduceSum(ib->Mul(grad_rhs, out), {-1});
    ShapeVector zeros2_shape = ib->GetShape(grad_rhs);
    if (zeros2_shape.size() > i1) {
      zeros2_shape[zeros2_shape.size() - i2] = 1;
    }
    auto zeros2 = ib->Zeros(ib->Value<ShapeVector>(zeros2_shape), ib->Value<int64_t>(ib->GetDtypeId(grad_rhs)));
    auto superdiag2 = ib->ReduceSum(
      ib->Mul(grad_rhs, ib->Concat({ib->StridedSlice(out, {{kLast2, {1, LLONG_MAX}}}), zeros2}, -k2)), {-1});
    auto subdiag2 =
      ib->ReduceSum(ib->Mul(grad_rhs, ib->Concat({zeros2, ib->StridedSlice(out, {{kLast2, {0, -1}}})}, -k2)), {-1});
    auto a = ib->Stack({superdiag2, diag2, subdiag2}, kLast2);
    grad_diags = ib->Sub(ib->Tensor(0, ib->GetDtype(a)), a);
  } else {
    grad_diags = ib->OutZeros(diagonals);
  }
  return {grad_diags, grad_rhs};
});

REG_BPROP_BUILDER("FFT").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "IFFT"); });
REG_BPROP_BUILDER("IFFT").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "FFT"); });
REG_BPROP_BUILDER("FFT2").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "IFFT2"); });
REG_BPROP_BUILDER("IFFT2").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "FFT2"); });
REG_BPROP_BUILDER("FFTN").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "IFFTN"); });
REG_BPROP_BUILDER("IFFTN").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "FFTN"); });

REG_BPROP_BUILDER("HFFT").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto n = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  auto n_type = n->abstract()->BuildType();
  if (n_type->isa<TypeNone>()) {
    n = ib->ShapeCalc(g_fft_axes_shape, {dout, dim}, {1})[0];
    n = ib->TupleGetItem(n, 0);
  }

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IHFFT", {dout, n, dim, norm});
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, dim});
  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("HFFT2").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &s = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IHFFT2", {dout, s, dim, norm});

  auto dim_n_res = ib->ShapeCalc(g_fft_cmpt_double_n_dim, {dout, dim}, {1})[0];
  auto n = ib->TupleGetItem(dim_n_res, 0);
  auto last_dim = ib->TupleGetItem(dim_n_res, 1);
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, last_dim});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("HFFTN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &s = ib->GetInput(i1);
  auto dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  // step1：Get the optional input.
  auto s_type = s->abstract()->BuildType();
  auto dim_type = dim->abstract()->BuildType();
  if (s_type->isa<TypeNone>() && dim_type->isa<TypeNone>()) {
    dim = ib->ShapeCalc(g_fft_shape_dim, {x})[1];
  }

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IHFFTN", {dout, s, dim, norm});

  auto dim_n_res = ib->ShapeCalc(g_fft_cmpt_double_n_dim, {dout, dim}, {1})[0];
  auto n = ib->TupleGetItem(dim_n_res, 0);
  auto last_dim = ib->TupleGetItem(dim_n_res, 1);
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, last_dim});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IHFFT").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto n = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto n_type = n->abstract()->BuildType();
  if (n_type->isa<TypeNone>()) {
    n = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
    n = ib->TupleGetItem(n, 0);
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFT", {dout, n, dim, norm});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IHFFT2").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto s_type = s->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    s = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
  }

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFT2", {dout, s, dim, norm});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IHFFTN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  auto dim = ib->GetInput(i2);
  const auto &norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto s_type = s->abstract()->BuildType();
  auto dim_type = dim->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    if (dim_type->isa<TypeNone>()) {
      dim = ib->ShapeCalc(g_fft_shape_dim, {x})[1];
    }
    s = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
  }

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFTN", {dout, s, dim, norm});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("RFFT").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto n = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);

  auto n_type = n->abstract()->BuildType();
  if (n_type->isa<TypeNone>()) {
    n = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
    n = ib->TupleGetItem(n, 0);
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFT", {dout, n, dim, norm});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("RFFT2").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);

  auto s_type = s->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    s = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFT2", {dout, s, dim, norm});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("RFFTN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  auto dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);

  auto s_type = s->abstract()->BuildType();
  auto dim_type = dim->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    if (dim_type->isa<TypeNone>()) {
      dim = ib->ShapeCalc(g_fft_shape_dim, {x})[1];
    }
    s = ib->ShapeCalc(g_fft_axes_shape, {x, dim}, {1})[0];
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("IFFTN", {dout, s, dim, norm});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  // step4：Return gradient results.
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IRFFT").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto n = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  auto n_type = n->abstract()->BuildType();
  if (n_type->isa<TypeNone>()) {
    n = ib->ShapeCalc(g_fft_axes_shape, {dout, dim}, {1})[0];
    n = ib->TupleGetItem(n, 0);
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("RFFT", {dout, n, dim, norm});
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, dim});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IRFFT2").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  auto s_type = s->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    s = ib->ShapeCalc(g_fft_axes_shape, {dout, dim}, {1})[0];
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("RFFT2", {dout, s, dim, norm});

  auto dim_n_res = ib->ShapeCalc(g_fft_cmpt_double_n_dim, {dout, dim}, {1})[0];
  auto n = ib->TupleGetItem(dim_n_res, 0);
  auto last_dim = ib->TupleGetItem(dim_n_res, 1);
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, last_dim});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IRFFTN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto s = ib->GetInput(i1);
  auto dim = ib->GetInput(i2);
  auto norm = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  norm = FFTNormReverse(ib, norm);
  auto s_type = s->abstract()->BuildType();
  auto dim_type = dim->abstract()->BuildType();
  if (s_type->isa<TypeNone>()) {
    if (dim_type->isa<TypeNone>()) {
      dim = ib->ShapeCalc(g_fft_shape_dim, {x})[1];
    }
    s = ib->ShapeCalc(g_fft_axes_shape, {dout, dim}, {1})[0];
  }
  // step2：Get the gradient.
  auto grad_dout = ib->Emit("RFFTN", {dout, s, dim, norm});

  auto dim_n_res = ib->ShapeCalc(g_fft_cmpt_double_n_dim, {dout, dim}, {1})[0];
  auto n = ib->TupleGetItem(dim_n_res, 0);
  auto last_dim = ib->TupleGetItem(dim_n_res, 1);
  grad_dout = ib->Emit("IRFFTDouble", {grad_dout, n, last_dim});

  auto x_shape = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
  grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape});

  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(s), ib->OutZeros(dim), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("FFTShift").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {ib->Emit("IFFTShift", {dout, dim}), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("IFFTShift").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &dim = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {ib->Emit("FFTShift", {dout, dim}), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("FFTFreq").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("RFFTFreq").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

DEF_PURE_SHAPE_CALC(g_correlate)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    constexpr int64_t input_num = 4;
    if (inputs.size() != input_num) {
      MS_LOG_EXCEPTION << "ShapeCalc[g_correlate] expect 4 inputs, but got " << inputs.size() << "inputs";
    }
    auto a_size = inputs.at(i0)[0];
    auto v_size = inputs.at(i1)[0];
    auto dout_size = inputs.at(i2)[0];
    auto mode_value = inputs.at(i3)[0];

    std::vector<int64_t> size_arr;
    size_arr.emplace_back(a_size + v_size - 1);
    std::vector<int64_t> begin_arr;
    begin_arr.emplace_back((a_size + v_size - dout_size) / 2);
    std::vector<int64_t> end_arr;
    end_arr.emplace_back((a_size + v_size - dout_size) / 2 + dout_size);

    constexpr int64_t same_mode = 1;
    if (mode_value == same_mode && a_size >= v_size && v_size % 2 == 0) {
      begin_arr[0] = begin_arr[0] - 1;
      end_arr[0] = end_arr[0] - 1;
    }
    return {size_arr, begin_arr, end_arr};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1, 1, 1}; });

REG_BPROP_BUILDER("Correlate").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto a = ib->GetInput(i0);
  auto v = ib->GetInput(i1);
  const auto &mode = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  // step1: if dtype of a/v is not float or complex, cast a, v dtyte as to dout dtype
  static const std::vector<TypeId> complex_or_float = {
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeComplex128,
  };
  auto a_type = ib->GetDtypeId(a);
  bool is_complex_or_float = std::any_of(complex_or_float.begin(), complex_or_float.end(),
                                         [&a_type](const TypeId &type_id) { return a_type == type_id; });
  if (!is_complex_or_float) {
    a = ib->Cast(a, ib->GetDtype(dout));
    v = ib->Cast(v, ib->GetDtype(dout));
  }

  // step2: pad dout to (a_size + v_size - 1)
  auto dout_padded = dout;
  int64_t mode_value = GetValue<int64_t>(mode->BuildValue());
  constexpr int64_t full_mode = 3;
  if (mode_value != full_mode) {
    // calculate StridedSliceGrad paragram [size] [begin] [end]
    auto param_array = ib->ShapeCalc(g_correlate, {a, v, dout, mode}, {3});
    dout_padded =
      ib->Emit("StridedSliceGrad",
               {dout, param_array[0], param_array[1], param_array[2], ib->Value<ShapeVector>(ShapeVector{1LL})},
               {{"begin_mask", MakeValue<int64_t>(0LL)},
                {"end_mask", MakeValue<int64_t>(0LL)},
                {"ellipsis_mask", MakeValue<int64_t>(0LL)},
                {"new_axis_mask", MakeValue<int64_t>(0LL)},
                {"shrink_axis_mask", MakeValue<int64_t>(0LL)}});
  }

  // step3: calculate da, dv by convolution1d, reverse, conj
  auto a_conj = a;
  if (ib->GetDtypeId(a) == kNumberTypeComplex64 || ib->GetDtypeId(a) == kNumberTypeComplex128) {
    a_conj = ib->Emit("Conj", {a});
  }
  auto v_r = ib->ReverseV2(v, ib->Value<ShapeVector>(ShapeVector{0LL}));
  auto dv_r = ib->Emit("Correlate", {dout_padded, a_conj, ib->Value<int64_t>(2LL)});
  auto da = ib->Emit("Correlate", {dout_padded, v_r, ib->Value<int64_t>(2LL)});
  auto dv_conj = ib->ReverseV2(dv_r, ib->Value<ShapeVector>(ShapeVector{0LL}));
  auto dv = dv_conj;
  if (ib->GetDtypeId(a) == kNumberTypeComplex64 || ib->GetDtypeId(a) == kNumberTypeComplex128) {
    dv = ib->Emit("Conj", {dv_conj});
  }

  return {da, dv, ib->OutZeros(mode)};
});

REG_BPROP_BUILDER("DCT").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  const auto &n = ib->GetInput(i2);
  auto n_type = n->abstract()->BuildType();
  const auto &axis = ib->GetInput(i3);
  auto norm = ib->GetInput(i4);
  auto dout = ib->GetInput(i6);

  auto x_shape_dim = ib->ShapeCalc(g_fft_shape_dim, {x});
  auto axis_to_tuple = ib->ShapeCalc(g_fft_dct_axis2tuple, {x, axis}, {1})[0];

  constexpr int64_t kNormBackward = 0;
  constexpr int64_t kNormOrtho = 2;
  auto norm_type = norm->abstract()->BuildType();
  if (norm_type->isa<TypeNone>()) {
    norm = ib->Value(kNormBackward);
  }
  auto norm_value = GetValue<int64_t>(norm->BuildValue());
  if (norm_value == kNormBackward) {
    dout = ib->Emit("FFTOrtho", {dout, axis_to_tuple, ib->Value(false)});
    norm = ib->Value(kNormOrtho);
  }

  auto grad_dout = ib->Emit("IDCT", {dout, type, n, axis, norm});
  if (!n_type->isa<TypeNone>()) {
    grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape_dim[0]});
  }
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(type), ib->OutZeros(n), ib->OutZeros(axis), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("DCTN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  const auto &s = ib->GetInput(i2);
  auto axes = ib->GetInput(i3);
  auto norm = ib->GetInput(i4);
  auto dout = ib->GetInput(i6);

  auto x_shape_dim = ib->ShapeCalc(g_fft_shape_dim, {x});

  auto s_type = s->abstract()->BuildType();
  auto dim_type = axes->abstract()->BuildType();
  if (dim_type->isa<TypeNone>()) {
    if (s_type->isa<TypeNone>()) {
      axes = x_shape_dim[1];
    } else {
      axes = ib->ShapeCalc(g_fft_dct_axes, {x, s}, {1})[0];
    }
  }

  constexpr int64_t kNormBackward = 0;
  constexpr int64_t kNormOrtho = 2;
  auto norm_type = norm->abstract()->BuildType();
  if (norm_type->isa<TypeNone>()) {
    norm = ib->Value(kNormBackward);
  }
  auto norm_value = GetValue<int64_t>(norm->BuildValue());
  if (norm_value == kNormBackward) {
    dout = ib->Emit("FFTOrtho", {dout, axes, ib->Value(false)});
    norm = ib->Value(kNormOrtho);
  }

  auto grad_dout = ib->Emit("IDCTN", {dout, type, s, axes, norm});
  if (!s_type->isa<TypeNone>()) {
    grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, x_shape_dim[0]});
  }
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }
  return {grad_dout, ib->OutZeros(type), ib->OutZeros(s), ib->OutZeros(axes), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IDCT").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  const auto &n = ib->GetInput(i2);
  const auto &axis = ib->GetInput(i3);
  const auto &norm = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("DCT", {dout, type, n, axis, norm});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto n_type = n->abstract()->BuildType();
  if (!n_type->isa<TypeNone>()) {
    auto res = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
    grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, res});
  }
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }

  // step4：Return gradient results.
  return {grad_dout, ib->OutZeros(type), ib->OutZeros(n), ib->OutZeros(axis), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("IDCTN").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &type = ib->GetInput(i1);
  const auto &s = ib->GetInput(i2);
  const auto &axes = ib->GetInput(i3);
  const auto &norm = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  // step2：Get the gradient.
  auto grad_dout = ib->Emit("DCTN", {dout, type, s, axes, norm});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto s_type = s->abstract()->BuildType();
  if (!s_type->isa<TypeNone>()) {
    auto res = ib->ShapeCalc(g_fft_shape_dim, {x})[0];
    grad_dout = ib->Emit("FFTShapeCopy", {grad_dout, res});
  }
  if (ib->GetDtypeId(x) != kNumberTypeBFloat16) {
    auto x_dtype = ib->GetDtype(x);
    grad_dout = ib->Cast(grad_dout, x_dtype);
  }

  // step4：Return gradient results.
  return {grad_dout, ib->OutZeros(type), ib->OutZeros(s), ib->OutZeros(axes), ib->OutZeros(norm)};
});

REG_BPROP_BUILDER("MeanExt").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &dtype = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);

  auto dx = MeanExtGrad(ib, input, axis, keep_dims, out, dout);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("Std").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &axis = ib->GetInput(i1);
  const auto &correction = ib->GetInput(i2);
  const auto &keep_dims = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);

  auto grad = StdGrad(ib, input, axis, correction, keep_dims, out, dout);
  return {grad, ib->OutZeros(axis), ib->OutZeros(correction), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("SumExt").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &dtype = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  NodePtr dx;
  auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
  if (!keep_dims_opt.has_value()) {
    auto true_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, true)}; };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, false)}; };
    auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
    dx = ib->Conditional(keep_dims_true, true_branch, false_branch);
  } else {
    dx = SumGrad(ib, input, axis, dout, keep_dims_opt.value());
  }

  return {ib->Cast(dx, ib->GetDtype(input)), ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("StdMean").SetUnusedInputs({}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &correction = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto std_dout = ib->TupleGetItem(dout, i0);
  auto mean_dout = ib->TupleGetItem(dout, i1);
  auto std_dout_defined = !std_dout->BuildValue()->isa<None>();
  auto mean_dout_defined = !mean_dout->BuildValue()->isa<None>();

  NodePtr dx = nullptr;
  if (std_dout_defined) {
    auto std_out = ib->TupleGetItem(out, i0);
    dx = StdGrad(ib, input, dim, correction, keepdim, std_out, std_dout);
  }

  if (mean_dout_defined) {
    auto mean_out = ib->TupleGetItem(out, i1);
    auto mean_grad = MeanExtGrad(ib, input, dim, keepdim, mean_out, mean_dout);
    dx = dx == nullptr ? mean_grad : ib->Add(dx, mean_grad);
  }
  return {dx, ib->OutZeros(dim), ib->OutZeros(correction), ib->OutZeros(keepdim)};
});

REG_BPROP_BUILDER("VarMean").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &correction = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto var_dout = ib->TupleGetItem(dout, i0);
  auto mean_dout = ib->TupleGetItem(dout, i1);
  auto var_dout_defined = !var_dout->BuildValue()->isa<None>();
  auto mean_dout_defined = !mean_dout->BuildValue()->isa<None>();

  NodePtr dx = nullptr;
  if (var_dout_defined) {
    dx = VarGrad(ib, input, dim, var_dout, correction, keepdim);
  }

  if (mean_dout_defined) {
    auto mean_out = ib->TupleGetItem(out, i1);
    auto mean_grad = MeanExtGrad(ib, input, dim, keepdim, mean_out, mean_dout);
    dx = dx == nullptr ? mean_grad : ib->Add(dx, mean_grad);
  }
  return {dx, ib->OutZeros(dim), ib->OutZeros(correction), ib->OutZeros(keepdim)};
});

REG_BPROP_BUILDER("Nansum").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &dtype = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  NodePtr dx;
  auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
  if (!keep_dims_opt.has_value()) {
    auto true_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, true)}; };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {SumGrad(e, input, axis, dout, false)}; };
    auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
    dx = ib->Conditional(keep_dims_true, true_branch, false_branch);
  } else {
    dx = SumGrad(ib, input, axis, dout, keep_dims_opt.value());
  }

  auto div_tensor = ib->LogicalNot(ib->NotEqual(input, input));
  dx = ib->Mul(dx, div_tensor);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(dtype)};
});

DEF_PURE_SHAPE_CALC(g_prod_ext)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(0);
    auto axis = inputs.at(1);
    if (axis.empty()) {
      axis.resize(input_shape.size());
      std::iota(axis.begin(), axis.end(), 0LL);
    }
    auto output_shape_kept_dims = ReduceShape(input_shape, axis);
    auto tile_scaling = ReduceShapeTupleDiv(input_shape, output_shape_kept_dims);
    auto [pack_shape, perm] = SplitShapeIndex(input_shape, axis);
    return {output_shape_kept_dims, tile_scaling, pack_shape, perm, InvertPermutation(perm)};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto input_shape = inputs.at(0);
    if (IsDynamicRank(input_shape) || !unknown_inputs.empty()) {
      return {-1, -1, -1, -1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, size, 2, size, size};
  });

REG_BPROP_BUILDER("ProdExt").SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto input_dtype = ib->GetDtype(input);
  auto input_dtype_id = ib->GetDtypeId(input);
  if (input_dtype_id == kNumberTypeComplex64 || input_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ProdExt', gradient not support for complex type currently.";
  }
  const auto &axis = ib->GetInput(i1);
  const auto &keep_dims = ib->GetInput(i2);
  const auto &dtype = ib->GetInput(i3);
  auto out = ib->GetInput(i4);
  auto out_dtype = ib->GetDtype(out);
  auto dout = ib->GetInput(i5);
  auto dout_dtype = ib->GetDtype(dout);
  auto dx = input;
  if (ib->GetRank(input) == 0) {
    return {dout, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(dtype)};
  }

  if (axis->abstract()->BuildType()->isa<TypeNone>()) {
    auto input_dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(input)));
    auto zero_input = ib->Zeros(ib->Shape(input), input_dtype);
    auto zero_mask = ib->Cast(ib->Equal(input, zero_input), kInt64);
    auto zero_num = ib->SumExt(zero_mask, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone));
    auto has_no_zero = ib->Equal(zero_num, ib->Tensor(static_cast<int64_t>(0)));
    auto has_one_more_zero = ib->Greater(zero_num, ib->Tensor(static_cast<int64_t>(1)));
    auto has_one_zero = ib->Equal(zero_num, ib->Tensor(static_cast<int64_t>(1)));

    auto all_false_branch = [&dx](const Emitter *e) -> NodePtrList { return {dx}; };
    auto no_zero_true_branch = [&](Emitter *e) -> NodePtrList {
      return {e->Cast(e->Mul(dout, e->Cast(e->Div(out, e->Cast(input, out_dtype)), dout_dtype)), input_dtype)};
    };
    dx = ib->Conditional(has_no_zero, no_zero_true_branch, all_false_branch);

    auto one_more_zero_true_branch = [&input](Emitter *e) -> NodePtrList {
      auto input_dtype_in_branch = e->Value(static_cast<int64_t>(input->dtype()->type_id()));
      return {e->Zeros(e->Shape(input), input_dtype_in_branch)};
    };
    dx = ib->Conditional(has_one_more_zero, one_more_zero_true_branch, all_false_branch);

    auto one_zero_true_branch = [&](Emitter *e) -> NodePtrList {
      auto res = e->ShapeCalc(g_prod_ext, {input, e->Value<std::vector<int64_t>>({})}, {1});

      auto grad = e->Tile(e->Reshape(dout, res[i0]), res[i1]);
      auto permuted = e->Transpose(input, res[i3]);
      auto permuted_shape = e->Shape(permuted);
      auto reshaped = e->Reshape(permuted, res[i2]);
      auto left = e->CumProd(reshaped, e->Value<int64_t>(0), true, false);
      auto right = e->CumProd(reshaped, e->Value<int64_t>(0), true, true);
      auto y = e->Reshape(e->Mul(left, right), permuted_shape);
      auto out = e->Mul(e->Transpose(y, res[i4]), grad);
      return {e->Cast(e->Reshape(out, e->Shape(input)), input_dtype)};
    };
    dx = ib->Conditional(has_one_zero, one_zero_true_branch, all_false_branch);
  } else {
    auto res = ib->ShapeCalc(g_prod_ext, {input, ib->MakeTuple({axis})}, {1});
    auto keep_dims_opt = mindspore::GetScalarValue<bool>(keep_dims->BuildValue());
    if (!keep_dims_opt.has_value()) {
      auto true_branch = [&out, &dout](Emitter *e) -> NodePtrList { return {out, dout}; };
      auto false_branch = [&out, &dout, &res](Emitter *e) -> NodePtrList {
        return {e->Reshape(out, res[i0]), e->Reshape(dout, res[i0])};
      };
      auto keep_dims_true = ib->Equal(keep_dims, ib->Value<bool>(true));
      auto cond_out = ib->Conditional(keep_dims_true, true_branch, false_branch);
      out = ib->TupleGetItem(cond_out, i0);
      dout = ib->TupleGetItem(cond_out, i1);
    } else {
      out = keep_dims_opt.value() ? out : ib->Reshape(out, res[i0]);
      dout = keep_dims_opt.value() ? dout : ib->Reshape(dout, res[i0]);
    }

    auto input_dtype = ib->Value(static_cast<int64_t>(ib->GetDtypeId(input)));
    auto zero_input = ib->Zeros(ib->Shape(input), input_dtype);
    auto zero_mask = ib->Equal(input, zero_input);
    auto zero_num = ib->SumExt(zero_mask, ib->EmitValue(kNone), ib->Value(false), ib->EmitValue(kNone));
    auto has_no_zero = ib->Equal(zero_num, ib->Tensor(static_cast<int64_t>(0)));

    auto no_zero_true_branch = [&](Emitter *e) -> NodePtrList {
      return {e->Cast(e->Mul(dout, e->Cast(e->Div(out, e->Cast(input, out_dtype)), dout_dtype)), input_dtype)};
    };
    auto no_zero_false_branch = [&](Emitter *e) -> NodePtrList {
      auto grad = e->Tile(dout, res[i1]);
      auto permuted = e->Transpose(input, res[i3]);
      auto permuted_shape = e->Shape(permuted);
      auto reshaped = e->Reshape(permuted, res[i2]);
      auto left = e->CumProd(reshaped, e->Value<int64_t>(0), true, false);
      auto right = e->CumProd(reshaped, e->Value<int64_t>(0), true, true);
      auto y = e->Reshape(e->Mul(left, right), permuted_shape);
      auto out = e->Mul(e->Transpose(y, res[i4]), grad);
      return {e->Cast(e->Reshape(out, e->Shape(input)), input_dtype)};
    };
    dx = ib->Conditional(has_no_zero, no_zero_true_branch, no_zero_false_branch);
  }

  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("Outer").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &vec2 = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  ShapeVector after_shape = {-1, 1};
  auto reshape_input = ib->Reshape(input, after_shape);
  NodePtr bc_dinput = nullptr;
  NodePtr bc_dvec2 = nullptr;
  if (input->need_compute_grad_out()) {
    bc_dinput = ib->Mul(vec2, dout);
  }
  if (vec2->need_compute_grad_out()) {
    bc_dvec2 = ib->Mul(reshape_input, dout);
  }
  auto grad = BinopGradCommon(ib, reshape_input, vec2, bc_dinput, bc_dvec2);
  grad[0] = ib->Reshape(grad[0], ib->Shape(input));
  return grad;
});

REG_BPROP_BUILDER("MultiScaleDeformableAttn").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &value = ib->GetInput(i0);
  const auto &shape = ib->GetInput(i1);
  const auto &offset = ib->GetInput(i2);
  const auto &locations_trans = ib->GetInput(i3);
  const auto &weight = ib->GetInput(i4);
  const auto &grad_output = ib->GetInput(i6);

  auto grad_out = ib->MultiScaleDeformableAttnGrad(value, shape, offset, locations_trans, weight, grad_output);
  auto grad0 = ib->TupleGetItem(grad_out, 0);
  auto grad1 = ib->TupleGetItem(grad_out, 1);
  auto grad2 = ib->TupleGetItem(grad_out, 2);

  return {grad0, ib->OutZeros(shape), ib->OutZeros(offset), grad1, grad2};
});

REG_BPROP_BUILDER("Dot").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &other = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr grad_input = nullptr;
  NodePtr grad_other = nullptr;
  if (input->need_compute_grad_out()) {
    auto other_type = ib->GetDtypeId(other);
    auto is_complex = other_type == kNumberTypeComplex64 || other_type == kNumberTypeComplex128;
    grad_input = ib->Mul(dout, is_complex ? ib->Emit("Conj", {other}) : other);
  }
  if (other->need_compute_grad_out()) {
    auto input_type = ib->GetDtypeId(input);
    auto is_complex = input_type == kNumberTypeComplex64 || input_type == kNumberTypeComplex128;
    grad_other = ib->Mul(dout, is_complex ? ib->Emit("Conj", {input}) : input);
  }
  return {grad_input, grad_other};
});

REG_BPROP_BUILDER("Var").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &correction = ib->GetInput(i2);
  const auto &keepdim = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);

  auto grad = VarGrad(ib, input, dim, dout, correction, keepdim);

  return {grad, ib->OutZeros(dim), ib->OutZeros(correction), ib->OutZeros(keepdim)};
});

REG_BPROP_BUILDER("Frac").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {dout};
});
REG_BPROP_BUILDER("BitwiseNot").FreeUselessValues_IO({i0}, {}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NsaCompress").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto weight = ib->GetInput(i1);
  auto compress_block_size = ib->GetInput(i2);
  auto compress_stride = ib->GetInput(i3);
  auto actual_seq_len = ib->GetInput(i4);
  auto grad_output = ib->GetInput(i6);

  auto grad_out =
    ib->Emit("NsaCompressGrad", {grad_output, input, weight, compress_block_size, compress_stride, actual_seq_len});
  auto grad_input = ib->TupleGetItem(grad_out, 0);
  auto grad_weight = ib->TupleGetItem(grad_out, 1);

  return {grad_input, grad_weight, ib->OutZeros(compress_block_size), ib->OutZeros(compress_stride),
          ib->OutZeros(actual_seq_len)};
});

REG_BPROP_BUILDER("NsaCompressAttention").SetUnusedInputs({i5, i6, i7, i8, i9, i13}).SetBody(BODYFUNC(ib) {
  auto query = ib->GetInput(i0);
  auto key = ib->GetInput(i1);
  auto value = ib->GetInput(i2);
  auto scale_value = ib->GetInput(i3);
  auto head_num = ib->GetInput(i4);
  auto atten_mask = ib->GetInput(i10);
  auto actual_seq_qlen = ib->GetInput(i11);
  auto actual_cmp_seq_kvlen = ib->GetInput(i12);

  // Forward output: [attention_out, topk_indices_out, softmax_max_out, softmax_sum_out]
  auto forward_output = ib->GetInput(i14);
  auto attention_out = ib->TupleGetItem(forward_output, 0);
  auto softmax_max_out = ib->TupleGetItem(forward_output, 2);
  auto softmax_sum_out = ib->TupleGetItem(forward_output, 3);

  auto dout = ib->GetInput(i15);
  auto grad_output = ib->TupleGetItem(dout, 0);

  // Call FlashAttentionScoreGrad for backward computation
  auto grad_out = ib->Emit("FlashAttentionScoreGrad", {
                                                        query,
                                                        key,
                                                        value,
                                                        grad_output,
                                                        ib->EmitValue(kNone),
                                                        ib->EmitValue(kNone),
                                                        ib->EmitValue(kNone),
                                                        atten_mask,
                                                        softmax_max_out,
                                                        softmax_sum_out,
                                                        ib->EmitValue(kNone),
                                                        attention_out,
                                                        ib->EmitValue(kNone),
                                                        actual_seq_qlen,
                                                        actual_cmp_seq_kvlen,
                                                        head_num,
                                                        ib->Value<float>(1.0),
                                                        scale_value,
                                                        ib->Value<int64_t>(2147483647),
                                                        ib->Value<int64_t>(2147483647),
                                                        ib->Value<int64_t>(0),
                                                        ib->Value<int64_t>(4),  // TND layout
                                                        ib->Value<int64_t>(1)   // sparse_mode
                                                      });

  auto grad_query = ib->TupleGetItem(grad_out, 0);
  auto grad_key = ib->TupleGetItem(grad_out, 1);
  auto grad_value = ib->TupleGetItem(grad_out, 2);
  auto grad_pse = ib->TupleGetItem(grad_out, 3);

  return {grad_query,
          grad_key,
          grad_value,
          ib->OutZeros(scale_value),
          ib->OutZeros(head_num),
          ib->OutZeros(ib->GetInput(i5)),
          ib->OutZeros(ib->GetInput(i6)),
          ib->OutZeros(ib->GetInput(i7)),
          ib->OutZeros(ib->GetInput(i8)),
          ib->OutZeros(ib->GetInput(i9)),
          ib->OutZeros(atten_mask),
          ib->OutZeros(actual_seq_qlen),
          ib->OutZeros(actual_cmp_seq_kvlen),
          ib->OutZeros(ib->GetInput(i13))};
});

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
