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
#include "ops/scatter_nd.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const std::vector<AbstractBasePtr> &input_args) {
  auto shape_value = input_args[kInputIndex2]->BuildValue();
  auto shape_value_element = GetValue<std::vector<int64_t>>(shape_value);
  for (const auto &shape : shape_value_element) {
    (void)CheckAndConvertUtils::CheckInteger("shape value", shape, kGreaterThan, 0, "ScatterNd");
  }
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto update_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("indices_shape[0] and update_shape[0]", indices_shape[0], kEqual,
                                           update_shape[0], "ScatterNd");
  return std::make_shared<abstract::Shape>(shape_value_element);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> indices_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> update_valid_types = {kTensorType};
  auto indices_type = input_args[0]->BuildType();
  auto update_type = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("update type", update_type, update_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indices_type, indices_valid_types, prim->name());
  return input_args[1]->BuildType();
}
}  // namespace

AbstractBasePtr ScatterNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args), InferShape(input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameScatterNd, ScatterNd);
}  // namespace ops
}  // namespace mindspore
