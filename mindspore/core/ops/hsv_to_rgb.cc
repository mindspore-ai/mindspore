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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const int64_t kNumDims = 4;
  const int64_t kLastDim = 3;
  const int64_t input_dims = SizeToLong(input_shape.size());
  const int64_t input_last_dims = SizeToLong(input_shape.cend()[-1]);
  CheckAndConvertUtils::CheckInteger("the dimension of [x]", input_dims, kEqual, kNumDims, kNameHSVToRGB);
  CheckAndConvertUtils::CheckInteger("the last dimension of the shape of [x]", input_last_dims, kEqual, kLastDim,
                                     kNameHSVToRGB);

  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> input_valid_types = {kFloat16, kFloat32, kFloat64};
  CheckAndConvertUtils::CheckTensorTypeValid("x", input_dtype, input_valid_types, kNameHSVToRGB);
  return input_dtype;
}
}  // namespace

AbstractBasePtr HSVToRGBInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = InferType(primitive, input_args);
  auto shapes = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(HSVToRGB, prim::kPrimHSVToRGB, HSVToRGBInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
