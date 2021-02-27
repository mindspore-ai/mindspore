/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <string>
#include "ops/mat_mul.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatMulInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("matmul_infer_input", input_args.size(), kEqual, 2, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShape("w_shape", input_args[1]->BuildShape(), prim_name);
  auto trans_a = GetValue<bool>(primitive->GetAttr(kTransposeA));
  auto trans_b = GetValue<bool>(primitive->GetAttr(kTransposeB));

  auto out_n = x_shape[0];
  auto out_m = w_shape[1];
  auto x_C = x_shape[1];
  auto w_C = w_shape[0];

  if (trans_a) {
    out_n = x_shape[1];
    x_C = x_shape[0];
  }
  if (trans_b) {
    out_m = w_shape[0];
    w_C = w_shape[1];
  }
  CheckAndConvertUtils::CheckInteger("dim C is not equal", x_C, kEqual, w_C, prim_name);
  primitive->AddAttr("transpose_x1", MakeValue(trans_a));
  primitive->AddAttr("transpose_x2", MakeValue(trans_b));
  std::vector<int64_t> out_shape = {out_n, out_m};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MatMulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,  kNumberTypeInt64,
                                        kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("w", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  if (infer_type == kNumberTypeInt8) {
    return std::make_shared<TensorType>(TypeIdToType(kNumberTypeInt32));
  }
  return TypeIdToType(infer_type);
}
}  // namespace

void MatMul::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}

void MatMul::set_transpose_a(bool transpose_a) { AddAttr(kTransposeA, MakeValue(transpose_a)); }

void MatMul::set_transpose_b(bool transpose_b) { AddAttr(kTransposeB, MakeValue(transpose_b)); }

bool MatMul::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMul::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

// Add
AbstractBasePtr MatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(MatMulInferType(primitive, input_args),
                                                    MatMulInferShape(primitive, input_args)->shape());
}

// Add
REGISTER_PRIMITIVE_C(kNameMatMul, MatMul);
}  // namespace ops
}  // namespace mindspore
