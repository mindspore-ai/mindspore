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

#include <set>
#include <algorithm>
#include "ops/arg_min_v2.h"
#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr ArgminV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape("ArgminV2", input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto x_shape_rank = SizeToLong(input_shape.size());
  ShapeVector out_shape = {};
  std::vector<int64_t> axis_value;
  int64_t axis_shape = 0;
  constexpr int dynamic_rank_value = -2;
  bool axis_is_dynamic = CheckAndGetAxisValue(input_args, &axis_value, &axis_shape, primitive);
  ReduceFuncCheckAxisInferImpl(primitive, &axis_value, input_shape.size());

  if ((x_shape_rank == 1 && input_shape[0] == dynamic_rank_value) || (axis_shape == -1)) {
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  if (axis_is_dynamic) {
    out_shape = ReduceFuncCalShapeAxisDyn(input_shape, axis_shape);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  out_shape = ReduceFuncCalShapeInferImpl(input_shape, axis_value);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ArgminV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  // ascend ArgMin supports float16 and float32.
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), x_valid_types, prim->name());
  const std::set<TypePtr> axis_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("axis", input_args[1]->BuildType(), axis_valid_types, prim->name());
  return kInt32;
}

MIND_API_OPERATOR_IMPL(ArgminV2, BaseOperator);
abstract::AbstractBasePtr ArgminV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  return abstract::MakeAbstract(ArgminV2InferShape(primitive, input_args), ArgminV2InferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(ArgminV2, prim::kPrimArgminV2, ArgminV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
