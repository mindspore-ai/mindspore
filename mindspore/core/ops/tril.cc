/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/tril.h"

#include <algorithm>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TrilInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  const int64_t kShapeSize = 2;
  auto input_shape = input_args[0];
  MS_EXCEPTION_IF_NULL(input_shape);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape->BuildShape());
  auto x_shape = shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto x_shape_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("x's rank", x_shape_rank, kGreaterEqual, kShapeSize, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr TrilInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto input_shape = input_args[0];
  MS_EXCEPTION_IF_NULL(input_shape);
  auto x_type = input_shape->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  std::set<TypePtr> valid_x_types(common_valid_types);
  (void)valid_x_types.emplace(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_x_types, prim_name);
  return x_type;
}
}  // namespace

void Tril::Init(const int64_t diagonal) { set_diagonal(diagonal); }

void Tril::set_diagonal(const int64_t diagonal) { (void)this->AddAttr(kDiagonal, api::MakeValue(diagonal)); }

int64_t Tril::get_diagonal() const { return GetValue<int64_t>(GetAttr(kDiagonal)); }

AbstractBasePtr TrilInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());

  auto infer_type = TrilInferType(primitive, input_args);
  auto infer_shape = TrilInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Tril, BaseOperator);

// AG means auto generated
class MIND_API AGTrilInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Tril, prim::kPrimTril, AGTrilInfer, false);
}  // namespace ops
}  // namespace mindspore
