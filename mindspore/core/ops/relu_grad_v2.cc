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
#include "ops/relu_grad_v2.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr const size_t kReluGradV2InputNum = 2;
constexpr const size_t kGradientIndex = 0;
constexpr const size_t kMaskIndex = 1;
constexpr const size_t kReluGradV2GradientDims = 4;
abstract::ShapePtr ReluGradV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto gradient_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGradientIndex]->BuildShape());
  auto gradient_input_shape = gradient_shape_map[kShape];
  if (gradient_input_shape.size() != kReluGradV2GradientDims) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The 'gradient' must be a 4-D tensor,but got a " +
                                  std::to_string(gradient_input_shape.size()) + "-D tensor";
  }
  auto mask_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kMaskIndex]->BuildShape());
  auto mask_input_shape = mask_shape_map[kShape];
  if (mask_input_shape.size() < kReluGradV2GradientDims) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The 'mask' dims must be greater than 4,but got " +
                                  std::to_string(mask_input_shape.size()) + "-D tensor";
  }
  auto gradient_build_shape = input_args[kGradientIndex]->BuildShape();
  MS_EXCEPTION_IF_NULL(gradient_build_shape);
  auto gradient_shape = gradient_build_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(gradient_shape);
  return gradient_shape;
}

TypePtr ReluGradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[kGradientIndex]);
  auto gradient_type = input_args[kGradientIndex]->BuildType();
  MS_EXCEPTION_IF_NULL(gradient_type);
  if (!gradient_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s "
                            << " input must be tensor type but got " << gradient_type->ToString();
  }
  return gradient_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ReluGradV2, BaseOperator);
AbstractBasePtr ReluGradV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  (void)CheckAndConvertUtils::CheckInteger("ReluGradV2 infer", input_args.size(), kEqual, kReluGradV2InputNum,
                                           primitive->name());
  auto type = ReluGradV2InferType(primitive, input_args);
  auto shape = ReluGradV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ReluGradV2, prim::kPrimReluGradV2, ReluGradV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
