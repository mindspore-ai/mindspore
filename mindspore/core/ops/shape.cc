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

#include "ops/shape.h"
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
AbstractBasePtr ShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  // infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), op_name);
  // infer type
  AbstractBasePtrList abs_list;
  std::transform(in_shape.begin(), in_shape.end(), std::back_inserter(abs_list),
                 [](int64_t item) -> std::shared_ptr<abstract::AbstractScalar> {
                   return std::make_shared<abstract::AbstractScalar>(item);
                 });
  auto abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  abs->set_value(MakeValue(in_shape));
  return abs;
}
REGISTER_PRIMITIVE_C(kNameShape, Shape);
}  // namespace ops
}  // namespace mindspore
