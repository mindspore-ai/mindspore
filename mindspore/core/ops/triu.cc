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
#include "ops/triu.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TriuInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
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

TypePtr TriuInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16,  kInt32,
                                         kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64, kBool};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return x_type;
}
}  // namespace

void Triu::Init(const int diagonal) { set_diagonal(diagonal); }

void Triu::set_diagonal(const int diagonal) { (void)this->AddAttr(kDiagonal, api::MakeValue(diagonal)); }

int Triu::get_diagonal() const { return static_cast<int>(GetValue<int64_t>(GetAttr(kDiagonal))); }

AbstractBasePtr TriuInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infertype = TriuInferType(primitive, input_args);
  auto infershape = TriuInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

MIND_API_OPERATOR_IMPL(Triu, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Triu, prim::kPrimTriu, TriuInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
