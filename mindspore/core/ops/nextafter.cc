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
#include <map>
#include <set>
#include <string>
#include "ops/nextafter.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NextAfterInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  MS_EXCEPTION_IF_NULL(primitive);
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr NextAfterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", input_args[0]->BuildType());
  (void)types.emplace("x2", input_args[1]->BuildType());
  auto x1_infer_type = input_args[0]->BuildType();
  auto x2_infer_type = input_args[1]->BuildType();
  const std::set<TypePtr> input_valid_types = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_infer_type, input_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_infer_type, input_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, input_valid_types, prim->name());
  return x1_infer_type;
}
}  // namespace

AbstractBasePtr NextAfterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = NextAfterInferType(primitive, input_args);
  auto infer_shape = NextAfterInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(NextAfter, BaseOperator);

// AG means auto generated
class MIND_API AGNextAfterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NextAfterInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NextAfterInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NextAfterInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NextAfter, prim::kPrimNextAfter, AGNextAfterInfer, false);
}  // namespace ops
}  // namespace mindspore
