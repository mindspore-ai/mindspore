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
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "ops/reciprocal.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ReciprocalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto reciprocal_prim = primitive->cast<PrimReciprocalPtr>();
  MS_EXCEPTION_IF_NULL(reciprocal_prim);
  auto prim_name = reciprocal_prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto in_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->GetShapeTrack(), prim_name);
  // infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> valid_x_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_type", x_type, valid_x_type, prim_name);
  return std::make_shared<abstract::AbstractTensor>(x_type, in_shape);
}
REGISTER_PRIMITIVE_C(kNameReciprocal, Reciprocal);
}  // namespace ops
}  // namespace mindspore
