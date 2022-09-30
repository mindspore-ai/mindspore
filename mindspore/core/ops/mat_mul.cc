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
#include <memory>
#include <vector>
#include "ops/mat_mul.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MatMul::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}

void MatMul::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, api::MakeValue(transpose_a)); }

void MatMul::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, api::MakeValue(transpose_b)); }

bool MatMul::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMul::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(MatMul, BaseOperator);
class MatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const auto prim_name = primitive->name();
    auto a_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(a_shape_ptr)[kShape];
    auto b_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto b_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(b_shape_ptr)[kShape];
    if (IsDynamicRank(a_shape) || IsDynamicRank(b_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }

    const int64_t mat_rank = 2;
    (void)CheckAndConvertUtils::CheckInteger("rank of a", SizeToLong(a_shape.size()), kEqual, mat_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of b", SizeToLong(b_shape.size()), kEqual, mat_rank, prim_name);

    const int dim0 = 0;
    const int dim1 = 1;
    bool transpose_a = GetValue<bool>(primitive->GetAttr(kTransposeA));
    bool transpose_b = GetValue<bool>(primitive->GetAttr(kTransposeB));
    auto a_real_shape = transpose_a ? std::vector<int64_t>{a_shape[dim1], a_shape[dim0]} : a_shape;
    auto b_real_shape = transpose_b ? std::vector<int64_t>{b_shape[dim1], b_shape[dim0]} : b_shape;

    if (!a_shape_ptr->IsDynamic() && !b_shape_ptr->IsDynamic() && a_real_shape[dim1] != b_real_shape[dim0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', after transpose if specified, the second dim of 'a' and the first dim of 'b'"
                               << " should be equal, but got " << a_real_shape[dim1] << " and " << b_real_shape[dim0]
                               << ".";
    }

    ShapeVector output_shape = std::vector<int64_t>{a_real_shape[dim0], b_real_shape[dim1]};

    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set valid_types = {kFloat16, kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64, kComplex64, kComplex128};
    std::map<std::string, TypePtr> types;
    auto a_type = input_args[kInputIndex0]->BuildType();
    (void)types.emplace("a", a_type);
    (void)types.emplace("b", input_args[kInputIndex1]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
    return a_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatMul, prim::kPrimMatMul, MatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
