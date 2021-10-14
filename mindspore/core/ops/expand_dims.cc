/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ExpandDimsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto dim_val = GetValue<int64_t>(input_args[1]->BuildValue());
  auto rank = x_shape.size();
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", dim_val, kIncludeBoth, {-SizeToLong(rank) - 1, rank}, prim_name);
  if (dim_val < 0) {
    dim_val += SizeToLong(x_shape.size()) + 1;
  }
  auto out_shape = x_shape;
  (void)out_shape.insert(out_shape.begin() + dim_val, 1, 1);

  // Infer type
  const int64_t x_index = 0;
  auto x_type = CheckAndConvertUtils::GetInputTensorType(input_args, x_index, prim_name);
  std::set<TypePtr> valid_x_type = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("x_type", x_type, valid_x_type, prim_name);
  return std::make_shared<abstract::AbstractTensor>(x_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameExpandDims, ExpandDims);
}  // namespace ops
}  // namespace mindspore
