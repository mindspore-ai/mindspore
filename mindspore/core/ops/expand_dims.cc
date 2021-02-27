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

#include <string>
#include <map>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/expand_dims.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/control_depend.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ExpandDimsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto expand_dims_prim = primitive->cast<PrimExpandDims>();
  MS_EXCEPTION_IF_NULL(expand_dims_prim);
  auto prim_name = expand_dims_prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto dim_val = GetValue<int64_t>(input_args[1]->BuildValue());
  auto rank = x_shape.size();
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", dim_val, kIncludeBoth, {-rank - 1, rank}, prim_name);
  if (dim_val < 0) {
    dim_val += x_shape.size() + 1;
  }
  auto out_shape = x_shape;
  out_shape.insert(out_shape.begin() + dim_val, 1, 1);

  // Infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> valid_x_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_type", x_type, valid_x_type, prim_name);
  return std::make_shared<abstract::AbstractTensor>(x_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameExpandDims, ExpandDims);
}  // namespace ops
}  // namespace mindspore
