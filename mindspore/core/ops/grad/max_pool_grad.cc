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

#include "ops/grad/max_pool_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaxPoolGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = std::make_shared<abstract::Shape>(shape_map[kShape], shape_map[kMinShape], shape_map[kMaxShape]);
  return shape;
}
TypePtr MaxPoolGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }

  return input_args[0]->BuildType();
}
}  // namespace
MIND_API_OPERATOR_IMPL(MaxPoolGrad, PoolGrad);
abstract::AbstractBasePtr MaxPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = MaxPoolGradInferType(primitive, input_args);
  auto infer_shape = MaxPoolGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MaxPoolGrad, prim::kPrimMaxPoolGrad, MaxPoolGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
