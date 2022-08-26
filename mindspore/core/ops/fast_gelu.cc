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

#include "ops/fast_gelu.h"

#include <algorithm>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr FastGeLUInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto x = input_args[0]->BuildShape();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr FastGeLUInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(FastGeLU, BaseOperator);
AbstractBasePtr FastGeLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = FastGeLUInferType(primitive, input_args);
  auto infer_shape = FastGeLUInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(FastGeLU, prim::kPrimFastGeLU, FastGeLUInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
