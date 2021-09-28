/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include <memory>
#include "ops/diag_part.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kScaleNum = 2;

abstract::ShapePtr DiagPartInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  if ((input_shape.size() % kScaleNum) != 0 || input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For DiagPart, input rank must be non-zero and even, but got rank "
                             << input_shape.size();
  }
  auto length = input_shape.size() / kScaleNum;
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < length; i++) {
    CheckAndConvertUtils::Check("input_shape[i + rank(input_shape) / 2]", input_shape[i + length], kEqual,
                                "input_shape[i]", input_shape[i], op_name, ValueError);
    (void)out_shape.emplace_back(input_shape[i]);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DiagPartInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_dtype = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("input type", x_dtype, common_valid_types, primitive->name());
}
}  // namespace
AbstractBasePtr DiagPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(DiagPartInferShape(primitive, input_args), DiagPartInferType(primitive, input_args));
}
}  // namespace ops
}  // namespace mindspore
