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

#include "ops/grad/max_pool_grad_grad_with_argmax.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaxPoolGradGradWithArgmaxInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_dim = 4;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("origin input shape size", SizeToLong(x_shape.size()), kEqual, input_dim,
                                           primitive->name());

  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("origin output shape size", SizeToLong(grad_shape.size()), kEqual, input_dim,
                                           primitive->name());

  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("grad shape size", SizeToLong(argmax_shape.size()), kEqual, input_dim,
                                           primitive->name());

  CheckAndConvertUtils::Check("argmax_shape", x_shape, kEqual, grad_shape, primitive->name(), ValueError);
  return std::make_shared<abstract::Shape>(argmax_shape);
}

TypePtr MaxPoolGradGradWithArgmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_index_types = {kInt32, kInt64};
  (void)types.emplace("argmax", input_args[kInputIndex2]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(types, valid_index_types, prim->name());

  types.clear();
  const std::set<TypePtr> valid_data_types = {kFloat16, kFloat32};
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("grad", input_args[kInputIndex1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_data_types, prim->name());
}
}  // namespace

AbstractBasePtr MaxPoolGradGradWithArgmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPoolGradGradWithArgmaxInferType(primitive, input_args);
  auto infer_shape = MaxPoolGradGradWithArgmaxInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}

MIND_API_OPERATOR_IMPL(MaxPoolGradGradWithArgmax, MaxPoolGradGrad);
REGISTER_PRIMITIVE_EVAL_IMPL(MaxPoolGradGradWithArgmax, prim::kPrimMaxPoolGradGradWithArgmax,
                             MaxPoolGradGradWithArgmaxInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
