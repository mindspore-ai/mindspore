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

#include "ops/unravel_index.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (!input_args[0]->isa<abstract::AbstractTensor>() || !input_args[1]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "Input must be a Tensor.";
  }

  auto op_name = primitive->name();
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto dims_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  // support dynamic shape
  if (IsDynamicRank(indices_shape) || IsDynamicRank(dims_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  const int64_t indices_shape_dim = SizeToLong(indices_shape.size());
  const int64_t dims_shape_dim = SizeToLong(dims_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("dims shape", dims_shape_dim, kEqual, 1, op_name);
  if (indices_shape_dim == 0) {
    int64_t dims_shape_v = dims_shape[0];
    std::vector<int64_t> output_shape;
    output_shape.push_back(dims_shape_v);
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("indices shape", indices_shape_dim, kEqual, 1, op_name);
    int64_t indices_shape_v = indices_shape[0];
    int64_t dims_shape_v = dims_shape[0];
    std::vector<int64_t> output_shape;
    output_shape.push_back(dims_shape_v);
    output_shape.push_back(indices_shape_v);
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr UravelIndexInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", input_args[0]->BuildType());
  (void)types.emplace("dims", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32, kInt64}, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(UnravelIndex, BaseOperator);
AbstractBasePtr UnravelIndexInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(UravelIndexInferType(primitive, input_args),
                                                    InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(UnravelIndex, prim::kPrimUnravelIndex, UnravelIndexInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
