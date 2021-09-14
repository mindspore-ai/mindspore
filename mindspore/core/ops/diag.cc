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
#include "ops/diag.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DiagInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input rank", SizeToLong(input_shape.size()), kGreaterEqual, 1L,
                                           primitive->name());
  std::vector<int64_t> out_shape(input_shape);
  (void)out_shape.insert(out_shape.end(), input_shape.begin(), input_shape.end());
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PartInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_dtype = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("input type", x_dtype, common_valid_types, primitive->name());
}
}  // namespace
AbstractBasePtr DiagInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(DiagInferShape(primitive, input_args), PartInferType(primitive, input_args));
}
}  // namespace ops
}  // namespace mindspore
