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

#include "ops/grad/random_gamma_grad.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr RandomGammaGradInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto alpha_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(alpha_shape_ptr);
  auto sample_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(sample_shape_ptr);

  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(alpha_shape_ptr)[kShape];
  auto sample_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(sample_shape_ptr)[kShape];

  const int64_t alpha_dim = SizeToLong(alpha_shape.size());
  const int64_t sample_dim = SizeToLong(sample_shape.size());
  const int64_t max_dim = 8;
  (void)CheckAndConvertUtils::CheckInteger("The dimension of alpha", alpha_dim, kLessThan, max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("The dimension of sample", sample_dim, kLessThan, max_dim, prim_name);
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr RandomGammaGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("alpha_type", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("sample_type", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);

  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(RandomGammaGrad, BaseOperator);
AbstractBasePtr RandomGammaGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto shape = RandomGammaGradInferShape(primitive, input_args);
  auto type = RandomGammaGradInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(RandomGammaGrad, prim::kPrimRandomGammaGrad, RandomGammaGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
