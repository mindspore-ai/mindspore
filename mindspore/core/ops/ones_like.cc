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
#include <string>
#include <vector>
#include <memory>
#include "ops/ones_like.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto OnesLike_prim = primitive->cast<PrimOnesLikePtr>();
  MS_EXCEPTION_IF_NULL(OnesLike_prim);
  auto prim_name = OnesLike_prim->name();
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  //  const std::set<TypeId> valid_types = {kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,  kNumberTypeInt64,
  //                                        kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,
  //                                        kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
  //                                        kNumberTypeBool};
  auto infer_type = input_args[0]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("infer_type", infer_type, common_valid_types, "OnesLike");
  return infer_type;
}
}  // namespace
AbstractBasePtr OnesLikeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameOnesLike, OnesLike);
}  // namespace ops
}  // namespace mindspore
