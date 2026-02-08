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
#include <optional>
#include <set>
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"
#include "backend/ms_backend/graph_fusion/expander/base/utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/op_enum.h"
#include "utils/value_utils.h"

namespace mindspore::graphkernel::expander {
constexpr int typeLevelBool = 0;
constexpr int typeLevelInt = 1;
constexpr int typeLevelReducedFloat = 2;
constexpr int typeLevelFloat = 3;
constexpr int typeLevelComplex = 4;

static inline int TypeToLevel(TypeId t) {
  static const std::unordered_map<TypeId, int> type_level_map = {
    {kNumberTypeBool, typeLevelBool},
    {kNumberTypeInt8, typeLevelInt},
    {kNumberTypeInt16, typeLevelInt},
    {kNumberTypeInt32, typeLevelInt},
    {kNumberTypeInt64, typeLevelInt},
    {kNumberTypeUInt8, typeLevelInt},
    {kNumberTypeUInt16, typeLevelInt},
    {kNumberTypeUInt32, typeLevelInt},
    {kNumberTypeUInt64, typeLevelInt},
    {kNumberTypeFloat16, typeLevelReducedFloat},
    {kNumberTypeBFloat16, typeLevelReducedFloat},
    {kNumberTypeFloat32, typeLevelFloat},
    {kNumberTypeFloat64, typeLevelFloat},
  };
  const auto it = type_level_map.find(t);
  return it != type_level_map.end() ? it->second : typeLevelComplex;
}

REG_EXPANDER_FUNC("AddN").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib, FormatDefaultNchwSame)) {
    return {};
  }
  // Check Inputs
  constexpr size_t min_inputs = 2;
  if (ib->inputs().size() < min_inputs) {
    MS_LOG(INFO) << "For 'AddN', the inputs num should be greater than 1, but got " << ib->inputs().size();
    return {};
  }

  auto result = ib->input(0);
  for (size_t i = 1; i < ib->inputs().size(); ++i) {
    result = ib->Add(result, ib->input(i));
  }
  return {result};
});

REG_EXPANDER_FUNC("AssignAdd").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  auto param = ib->input(0);
  auto x = ib->input(1);
  if (x->GetDtype() != param->GetDtype()) {
    x = ib->Cast(x, param->GetDtype());
  }
  auto result = ib->Assign(param, ib->Add(param, x));
  return {result};
});

REG_EXPANDER_FUNC("Tanh").SetBody(BODYFUNC(ib) {
  if (ib->input(kIndex0)->GetDtype() != TypeIdToType(kNumberTypeFloat32)) {
    MS_LOG(DEBUG) << "For Tanh expander, input must be float32";
    return {};
  }
  auto result = ib->Tanh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Sinh").SetBody(BODYFUNC(ib) {
  if (ib->input(kIndex0)->GetDtype() != TypeIdToType(kNumberTypeFloat32)) {
    MS_LOG(DEBUG) << "For Sinh expander, input must be float32";
    return {};
  }
  auto result = ib->Sinh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Cosh").SetBody(BODYFUNC(ib) {
  if (ib->input(kIndex0)->GetDtype() != TypeIdToType(kNumberTypeFloat32)) {
    MS_LOG(DEBUG) << "For Cosh expander, input must be float32";
    return {};
  }
  auto result = ib->Cosh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("LogicalXor").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  const auto &input_y = ib->input(kIndex1);

  auto result_b = ib->LogicalAnd(input_x, ib->LogicalNot(input_y));
  auto result_a = ib->LogicalAnd(input_y, ib->LogicalNot(input_x));
  return {ib->LogicalOr(result_a, result_b)};
});

NodePtrList FastGeluExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(-val, input_x->GetDtype());
  auto const_1 = ib->Tensor(val / 2, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto sub = ib->Sub(input_x, abs);
  auto exp_0 = ib->Exp(ib->Mul(const_1, sub));
  auto n = ib->Mul(input_x, exp_0);
  auto exp_1 = ib->Exp(ib->Mul(const_0, abs));
  auto d = ib->Add(exp_1, const_2);

  return {ib->Div(n, d)};
}

NodePtrList FastGeluGradExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex1);
  const auto &dout = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(val, input_x->GetDtype());
  auto const_1 = ib->Tensor(-val, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto mul_1 = ib->Exp(ib->Mul(const_1, abs));
  auto mul_3 = ib->Mul(input_x, mul_1);
  mul_3 = ib->Mul(const_0, mul_3);
  mul_3 = ib->Add(mul_3, mul_1);

  auto sub = ib->Sub(input_x, abs);
  sub = ib->Exp(ib->Mul(sub, const_0));

  mul_3 = ib->Add(sub, mul_3);
  mul_1 = ib->Add(mul_1, const_2);
  mul_1 = ib->Mul(mul_1, mul_1);

  return {ib->Mul(ib->Div(mul_3, mul_1), dout)};
}

REG_EXPANDER_FUNC("FastGelu").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeLU").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeluGrad").SetBody(FastGeluGradExpand);
REG_EXPANDER_FUNC("FastGeLUGrad").SetBody(FastGeluGradExpand);

REG_EXPANDER_FUNC("SiLU").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto result = ib->Div(input_x, add_exp);
  return {result};
});

REG_EXPANDER_FUNC("SiLUGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->input(kIndex1);
  auto dout = ib->input(kIndex0);
  auto x_type = input_x->GetDtype()->type_id();
  auto need_cast = x_type == kNumberTypeFloat16;
  if (need_cast) {
    input_x = ib->Cast(input_x, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto sigmod = ib->Div(const_one, add_exp);
  auto out = ib->Div(input_x, add_exp);

  auto result = ib->Sub(ib->Add(sigmod, out), ib->Mul(sigmod, out));
  result = ib->Mul(result, dout);
  if (need_cast) {
    result = ib->Cast(result, x_type);
  }
  return {result};
});

REG_EXPANDER_FUNC("Addcmul").SetBody(BODYFUNC(ib) {
  if (!CheckAllDataTypeSame(ib)) {
    return {};
  }
  const auto &input_data = ib->input(kIndex0);
  const auto &x1 = ib->input(kIndex1);
  const auto &x2 = ib->input(kIndex2);
  const auto &value = ib->input(kIndex3);
  auto result = ib->Add(input_data, ib->Mul(value, ib->Mul(x1, x2)));
  return {result};
});

REG_EXPANDER_FUNC("ReluGrad").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  const auto &input_dout = ib->input(kIndex0);
  const auto &input_x = ib->input(kIndex1);
  auto const_zero = ib->Tensor(0.0, input_x->GetDtype());
  auto gt_res = ib->Greater(input_x, const_zero);
  auto res = ib->Select(gt_res, input_dout, const_zero);
  return {res};
});

REG_EXPANDER_FUNC("DivMod").SetBody(BODYFUNC(ib) {
  auto input1 = ib->input(kIndex0);
  auto input2 = ib->input(kIndex1);
  auto rounding_mode = ib->input(kIndex2);
  if (rounding_mode->GetDtype() == TypeIdToType(kMetaTypeNone)) {
    return {ib->Div(input1, input2)};
  }
  auto rounding_mode_value = GetValue<int64_t>(rounding_mode->GetValue());
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto out_type = input1->GetDtype();
  auto floor_f16 = rounding_mode_value == ops::RoundingMode::FLOOR && out_type == TypeIdToType(kNumberTypeFloat16);
  if (!(out_type == f32) && !floor_f16) {
    input1 = ib->Cast(input1, f32);
    input2 = ib->Cast(input2, f32);
  }
  auto result = ib->Div(input1, input2);
  result = floor_f16 ? ib->Cast(result, f32) : result;
  if (rounding_mode_value == ops::RoundingMode::FLOOR) {
    result = ib->Floor(result);
  } else if (rounding_mode_value == ops::RoundingMode::TRUNC) {
    if (out_type == TypeIdToType(kNumberTypeBFloat16) || out_type == TypeIdToType(kNumberTypeFloat16)) {
      result = ib->Cast(result, out_type);
      result = ib->Cast(result, f32);
    }
    result = ib->Trunc(result);
  } else {
    MS_LOG(DEBUG) << "rounding_mode is not supported: " << rounding_mode_value;
    return {};
  }
  result = out_type == f32 ? result : ib->Cast(result, out_type);
  return {result};
});

enum ReduceType { kSum = 0, kMean };

std::optional<std::vector<int64_t>> GetAxis(const NodePtr &axis) {
  std::vector<int64_t> res;
  if (axis->GetDtype()->type_id() != kMetaTypeNone) {
    auto axis_value = axis->GetValue();
    bool is_valid_axis =
      axis_value->isa<ValueSequence>() || axis_value->isa<tensor::Tensor>() || axis_value->isa<Scalar>();
    if (!is_valid_axis) {
      return std::nullopt;
    }
    res = GetAxisList(axis->GetValue());
  }
  return res;
}

NodePtrList ReduceExtCommon(const DefaultIrBuilder *ib, ReduceType reduce_type) {
  auto input = ib->input(kIndex0);
  auto input_type = input->GetDtype()->type_id();
  if (input_type != kNumberTypeFloat32 && input_type != kNumberTypeFloat16 && input_type != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "In MeanExt, dtype must be float16 or float32 or bfloat16";
    return {};
  }
  auto axis = ib->input(kIndex1);
  auto keep_dims = ib->input(kIndex2);
  auto dtype = ib->input(kIndex3);
  auto out_type = input_type;
  if (dtype->GetDtype() != TypeIdToType(kMetaTypeNone)) {
    out_type = static_cast<TypeId>(GetValue<int64_t>(dtype->GetValue()));
  }
  input = input_type == kNumberTypeFloat32 ? input : ib->Cast(input, kNumberTypeFloat32);
  auto x_shape = input->GetShape();
  if (x_shape.empty() || IsDynamicRank(x_shape)) {
    MS_LOG(DEBUG) << "Skip expanding node, bucause shape of this node is empty or dynamic rank, shape is: " << x_shape;
    return {};
  }
  auto axis_opt = GetAxis(axis);
  if (!axis_opt.has_value()) {
    return {};
  }
  std::vector<int64_t> axis_ = axis_opt.value();
  auto rank = SizeToLong(x_shape.size());
  (void)std::for_each(axis_.begin(), axis_.end(), [rank](auto &a) { a = a < 0 ? a + rank : a; });
  std::set<int64_t> uniq_axis(axis_.begin(), axis_.end());
  if (uniq_axis.size() != axis_.size()) {
    // duplicate axis not supported in SumExt/MeanExt
    MS_LOG(DEBUG) << "Duplicate value found in reduce axis: " << axis_;
    return {};
  }
  if (axis_.empty()) {
    for (int64_t i = 0; i < rank; ++i) {
      axis_.push_back(i);
    }
  }
  for (const auto &a : axis_) {
    if (x_shape.at(LongToSize(a)) < 0) {
      MS_LOG(DEBUG) << "Input shape " << x_shape << " at reduce axis [" << a << "] is dynamic";
      return {};
    }
  }
  int64_t sz = 1;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (std::find(axis_.begin(), axis_.end(), SizeToLong(i)) != axis_.end()) {
      sz *= x_shape[i];
    }
  }
  auto res = ib->ReduceSum(input, axis_, GetValue<bool>(keep_dims->GetValue()));
  if (reduce_type == ReduceType::kMean) {
    res = ib->Div(res, sz);
  }
  res = out_type == kNumberTypeFloat32 ? res : ib->Cast(res, out_type);
  return {res};
}

REG_EXPANDER_FUNC("SumExt").SetBody(BODYFUNC(ib) { return ReduceExtCommon(ib, ReduceType::kSum); });

REG_EXPANDER_FUNC("MeanExt").SetBody(BODYFUNC(ib) { return ReduceExtCommon(ib, ReduceType::kMean); });

REG_EXPANDER_FUNC("AcoshExt").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != TypeIdToType(kNumberTypeFloat32);
  auto cast = need_cast ? ib->Cast(input, f32) : input;
  auto mul = ib->Add(ib->Mul(cast, cast), -1);
  auto ln1 = ib->Log(ib->Add(ib->Sqrt(mul), cast));
  auto ln2 = ib->Add(ib->Log(cast), 6.931472e-01f);
  auto res = ib->Minimum(ln1, ln2);
  res = need_cast ? ib->Cast(res, input->GetDtype()) : res;
  return {res};
});

REG_EXPANDER_FUNC("AsinhExt").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != f32;
  auto cast = need_cast ? ib->Cast(input, f32) : input;
  auto abs = ib->Abs(cast);
  auto mul = ib->Add(ib->Mul(abs, abs), 1);
  auto add = ib->Add(ib->Sqrt(mul), abs);
  auto res = ib->Log(add);
  auto min = ib->Neg(res);
  auto ge = ib->Greater(cast, ib->Tensor(0, f32));
  res = ib->Select(ge, res, min);
  res = need_cast ? ib->Cast(res, input->GetDtype()) : res;
  return {res};
});

REG_EXPANDER_FUNC("Erf").SetBody(BODYFUNC(ib) {
  const auto &x = ib->input(kIndex0);
  const float a1 = 5.3443748503923416e-02f;
  const float a2 = 7.5517015457153320e+00f;
  const float a3 = 1.0162808990478516e+02f;
  const float a4 = 1.3938061523437500e+03f;
  const float a5 = 5.0637915039062500e+03f;
  const float a6 = 2.9639384765625000e+04f;
  const float b1 = 3.1212858200073242e+01f;
  const float b2 = 3.9856964111328125e+02f;
  const float b3 = 3.0231247558593750e+03f;
  const float b4 = 1.3243366210937500e+04f;
  const float b5 = 2.6267224609375000e+04f;
  const float t = 3.9200000762939453e+00f;
  // when x belongs to [-t,t],  Erf(x) = ((((((a1*x^2 + a2)*x^2 + x3)*x^2 + a4)*x^2 + a5)*x^2 + a6) * x)
  //                                   / (((((x^2 + b1)*x^2 + b2)*x^2 + b3)*x^2 + b4)*x^2 + b5)
  // else, Erf(x) = 1 or -1
  auto need_cast = x->GetDtype() != TypeIdToType(kNumberTypeFloat32);
  auto cast_1 = need_cast ? ib->Cast(x, kNumberTypeFloat32) : x;
  cast_1 = ib->Minimum(cast_1, t);
  cast_1 = ib->Maximum(cast_1, -t);
  auto mul_2 = ib->Mul(cast_1, cast_1);
  auto mul_3 = ib->Mul(mul_2, a1);
  mul_3 = ib->Add(mul_3, a2);
  mul_3 = ib->Mul(mul_3, mul_2);
  mul_3 = ib->Add(mul_3, a3);
  mul_3 = ib->Mul(mul_3, mul_2);
  mul_3 = ib->Add(mul_3, a4);
  mul_3 = ib->Mul(mul_3, mul_2);
  mul_3 = ib->Add(mul_3, a5);
  mul_3 = ib->Mul(mul_3, mul_2);
  mul_3 = ib->Add(mul_3, a6);
  mul_3 = ib->Mul(mul_3, cast_1);
  auto add_4 = ib->Add(mul_2, b1);
  add_4 = ib->Mul(add_4, mul_2);
  add_4 = ib->Add(add_4, b2);
  add_4 = ib->Mul(add_4, mul_2);
  add_4 = ib->Add(add_4, b3);
  add_4 = ib->Mul(add_4, mul_2);
  add_4 = ib->Add(add_4, b4);
  add_4 = ib->Mul(add_4, mul_2);
  add_4 = ib->Add(add_4, b5);
  auto result = ib->Div(mul_3, add_4);
  result = need_cast ? ib->Cast(result, x->GetDtype()) : result;
  return {result};
});

NodePtrList BinaryExtCommon(const DefaultIrBuilder *ib, bool is_add) {
  auto x0 = ib->input(kIndex0);
  auto x1 = ib->input(kIndex1);
  auto alpha = ib->input(kIndex2);
  auto x0_type = x0->GetDtype()->type_id();
  auto x1_type = x1->GetDtype()->type_id();
  auto alpha_type = alpha->GetDtype()->type_id();
  static std::set<TypeId> dvm_supported_types{kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeInt32,
                                              kNumberTypeBFloat16};
  if (x0_type == kNumberTypeBool) {
    MS_LOG(DEBUG) << "The data type of first input is not supported: " << TypeIdToString(x0_type);
    return {};
  }
  if (x1_type != x0_type) {
    MS_LOG(DEBUG) << "x0 and x1 have different data type: " << TypeIdToString(x0_type) << " vs "
                  << TypeIdToString(x1_type);
    return {};
  }
  if (alpha_type != x0_type) {
    if (x0_type == kNumberTypeBFloat16 && alpha_type == kNumberTypeFloat32) {
      // in this case, do not cast alpha because input tensors will be cast to fp32 later,
      // cast alpha to bf16 here and back to fp32 later will cause precision reduction
      x0 = ib->Cast(x0, alpha_type);
      x1 = ib->Cast(x1, alpha_type);
    } else {
      alpha = ib->ScalarToTensor(alpha, x0->GetDtype());
      auto alpha_value = alpha->GetValue();
      if (GraphKernelFlags::GetInstance().kernel_generator == "DVM" &&
          (alpha_value == nullptr || !IsValueKnown(alpha_value)) &&
          dvm_supported_types.find(alpha_type) == dvm_supported_types.end()) {
        MS_LOG(DEBUG) << "alpha is not const value and the data type of it is not supported: "
                      << TypeIdToString(alpha_type);
        return {};
      }
    }
  }
  x1 = ib->Mul(x1, alpha);
  auto result = is_add ? ib->Add(x0, x1) : ib->Sub(x0, x1);
  if (result->GetDtype()->type_id() != x0_type) {
    result = ib->Cast(result, x0_type);
  }
  return {result};
}

REG_EXPANDER_FUNC("AddExt").SetBody(BODYFUNC(ib) { return BinaryExtCommon(ib, true); });

REG_EXPANDER_FUNC("SubExt").SetBody(BODYFUNC(ib) { return BinaryExtCommon(ib, false); });

REG_EXPANDER_FUNC("Muls").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto scalar = ib->input(kIndex1);
  auto input_type = input->GetDtype();
  auto scalar_type = scalar->GetDtype();
  auto scalar_value = scalar->GetValue();
  if (scalar_value == nullptr || !IsValueKnown(scalar_value)) {
    MS_LOG(DEBUG) << "scalar is not const value and the data type of it is not supported: "
                  << TypeIdToString(scalar_type->type_id());
    return {};
  }
  auto dst_type = TypeToLevel(input_type->type_id()) >= TypeToLevel(scalar_type->type_id()) ? input_type : scalar_type;
  scalar = ib->ScalarToTensor(scalar, dst_type);
  if (input_type != dst_type) {
    input = ib->Cast(input, dst_type);
  }
  return {ib->Mul(input, scalar)};
});

REG_EXPANDER_FUNC("TanhGrad").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  auto input_y = ib->input(kIndex0);
  auto input_dy = ib->input(kIndex1);
  auto y_type = input_y->GetDtype();
  auto dy_type = input_dy->GetDtype();
  if (y_type->type_id() != dy_type->type_id()) {
    return {};
  }
  // for reduced floating types like fp16 and bf16, cast is required to guarantee the precision is acceptable
  if (y_type->type_id() == kNumberTypeFloat16 || y_type->type_id() == kNumberTypeBFloat16) {
    auto fp32 = TypeIdToType(kNumberTypeFloat32);
    auto input_y_fp32 = ib->Cast(input_y, fp32);
    auto input_dy_fp32 = ib->Cast(input_dy, fp32);
    auto const_one_fp32 = ib->Tensor(1, fp32);
    auto output_fp32 = ib->Mul(input_dy_fp32, ib->Sub(const_one_fp32, ib->Mul(input_y_fp32, input_y_fp32)));
    return {ib->Cast(output_fp32, y_type)};
  } else {
    auto const_one = ib->Tensor(1, y_type);
    return {ib->Mul(input_dy, ib->Sub(const_one, ib->Mul(input_y, input_y)))};
  }
});
}  // namespace mindspore::graphkernel::expander
