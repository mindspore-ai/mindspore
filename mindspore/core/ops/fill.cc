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

#include "ops/fill.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr FillInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_dtype = input_args[0]->cast<abstract::AbstractTypePtr>();
  MS_EXCEPTION_IF_NULL(input_dtype);
  auto dtype_value = input_dtype->BuildValue();
  MS_EXCEPTION_IF_NULL(dtype_value);
  auto dtype = dtype_value->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(dtype);
  auto valid_types = common_valid_types;
  valid_types.insert(kNumberTypeBool);
  CheckAndConvertUtils::CheckTypeSame("output datatype", dtype, valid_types, prim_name);
  auto out_shape = GetValue<std::vector<int64_t>>(input_args[1]->BuildValue());
  return std::make_shared<abstract::AbstractTensor>(dtype, std::make_shared<abstract::Shape>(out_shape));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Fill, prim::kPrimFill, FillInfer);
REGISTER_PRIMITIVE_C(kNameFill, Fill);
}  // namespace ops
}  // namespace mindspore
