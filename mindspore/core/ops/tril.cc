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
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TrilInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  const int64_t kShapeSize = 2;
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
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

  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
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
REGISTER_PRIMITIVE_EVAL_IMPL(Tril, prim::kPrimTril, TrilInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
