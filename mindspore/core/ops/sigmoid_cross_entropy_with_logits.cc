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

#include "ops/sigmoid_cross_entropy_with_logits.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SigmoidCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto sigmoid_prim = primitive->cast<PrimSigmoidCrossEntropyWithLogitsPtr>();
  MS_EXCEPTION_IF_NULL(sigmoid_prim);
  auto prim_name = sigmoid_prim->name();
  CheckAndConvertUtils::CheckInteger("sigmoid_cross_extropy_with_logits_infer", input_args.size(), kEqual, 2,
                                     prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShape("y_shape", input_args[1]->BuildShape(), prim_name);
  CheckAndConvertUtils::Check("x_shape", x_shape, kEqual, "y_shape", y_shape, prim_name, TypeError);

  // Infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  const std::set<TypeId> valid_types = {
    kNumberTypeBool,    kNumberTypeInt,     kNumberTypeInt8,    kNumberTypeInt16,
    kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt,    kNumberTypeUInt8,
    kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,  kNumberTypeFloat,
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64};
  std::map<std::string, TypePtr> args;
  args.emplace("x_type", input_args[0]->BuildType());
  args.emplace("y_type", input_args[1]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  return std::make_shared<abstract::AbstractTensor>(x_type, x_shape);
}
REGISTER_PRIMITIVE_C(kNameSigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits);
}  // namespace ops
}  // namespace mindspore
