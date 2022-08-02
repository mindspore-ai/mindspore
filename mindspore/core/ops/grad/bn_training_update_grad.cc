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

#include "ops/grad/bn_training_update_grad.h"

#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBNTrainingUpdateGradInputNum = 4;

abstract::TupleShapePtr BNTrainingUpdateGradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateGradInputNum, prim_name);
  auto batch_mean_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto batch_variance_shape_ptr = input_args[kInputIndex3]->BuildShape();
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{batch_mean_shape_ptr, batch_variance_shape_ptr});
}

TuplePtr BNTrainingUpdateGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateGradInputNum, prim_name);
  auto batch_mean_type_ptr = input_args[kInputIndex2]->BuildType();
  auto batch_variance_type_ptr = input_args[kInputIndex3]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{batch_mean_type_ptr, batch_variance_type_ptr});
}
}  // namespace

MIND_API_OPERATOR_IMPL(BNTrainingUpdateGrad, BaseOperator);
AbstractBasePtr BNTrainingUpdateGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BNTrainingUpdateGradInferShape(primitive, input_args),
                                BNTrainingUpdateGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BNTrainingUpdateGrad, prim::kPrimBNTrainingUpdateGrad, BNTrainingUpdateGradInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
