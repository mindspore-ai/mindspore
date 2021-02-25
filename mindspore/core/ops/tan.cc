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

#include "ops/tan.h"
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
AbstractBasePtr TanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto tan_prim = primitive->cast<PrimTanPtr>();
  MS_EXCEPTION_IF_NULL(tan_prim);
  auto prim_name = tan_prim->name();
  CheckAndConvertUtils::CheckInteger("tan_infer", input_args.size(), kEqual, 1, prim_name);

  // Infer Shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto infer_shape = std::make_shared<abstract::Shape>(x_shape);

  // Infer Type
  auto dtype = input_args[0]->BuildType();
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeInt32};
  CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", dtype, valid_types, prim_name);
  auto tensor_type = dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  auto infer_type = std::make_shared<TensorType>(TypeIdToType(element->type_id()));

  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
REGISTER_PRIMITIVE_C(kNameTan, Tan);
}  // namespace ops
}  // namespace mindspore
