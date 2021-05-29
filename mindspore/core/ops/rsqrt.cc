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

#include "ops/rsqrt.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/infer_base.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto rsqrt_prim = primitive->cast<PrimRsqrtPtr>();
  MS_EXCEPTION_IF_NULL(rsqrt_prim);
  auto prim_name = rsqrt_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("in_shape", input_args[0]->GetShapeTrack(), prim_name);
  CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(in_shape.size()), kEqual, 1, prim_name);
  return std::make_shared<abstract::Shape>(in_shape);
}
}  // namespace

AbstractBasePtr RsqrtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  size_t input_num = 1;
  auto type = InferBase::CheckSameInferType(primitive, input_args, common_valid_types, input_num);
  return std::make_shared<abstract::AbstractTensor>(type, InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameRsqrt, Rsqrt);
}  // namespace ops
}  // namespace mindspore
