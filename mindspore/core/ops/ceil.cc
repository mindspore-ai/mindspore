/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <memory>
#include <vector>

#include "ops/ceil.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CeilInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  const int64_t max_dim = 8;
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("The dimension of Ceil input", SizeToLong(in_shape.size()), kLessThan,
                                           max_dim, prim_name);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}
TypePtr CeilInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  const std::set valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Ceil, BaseOperator);
AbstractBasePtr CeilInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto type = CeilInferType(primitive, input_args);
  auto shape = CeilInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGCeilInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CeilInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CeilInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CeilInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Ceil, prim::kPrimCeil, AGCeilInfer, false);
}  // namespace ops
}  // namespace mindspore
