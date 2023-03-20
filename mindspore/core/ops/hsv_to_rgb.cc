/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "ops/hsv_to_rgb.h"

#include <set>
#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr HSVToRGBInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &build_shape = input_args[0]->BuildShape();
  if (build_shape->IsDimZero()) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of input can not be empty.";
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(build_shape)[kShape];
  // dynamic rank
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // dynamic shape
  if (IsDynamic(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector(abstract::Shape::kShapeDimAny, input_shape.size()));
  }
  const int64_t kNumDims = 4;
  const int64_t kLastDim = 3;
  const int64_t input_dims = SizeToLong(input_shape.size());
  const int64_t input_last_dims = input_shape.cend()[-1];
  (void)CheckAndConvertUtils::CheckInteger("the dimension of [x]", input_dims, kEqual, kNumDims, kNameHSVToRGB);
  (void)CheckAndConvertUtils::CheckInteger("the last dimension of the shape of [x]", input_last_dims, kEqual, kLastDim,
                                           kNameHSVToRGB);

  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr HSVToRGBInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> input_valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_dtype, input_valid_types, kNameHSVToRGB);
  return input_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(HSVToRGB, BaseOperator);
AbstractBasePtr HSVToRGBInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = HSVToRGBInferType(primitive, input_args);
  auto shapes = HSVToRGBInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGHSVToRGBInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return HSVToRGBInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return HSVToRGBInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return HSVToRGBInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(HSVToRGB, prim::kPrimHSVToRGB, AGHSVToRGBInfer, false);
}  // namespace ops
}  // namespace mindspore
