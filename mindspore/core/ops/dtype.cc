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

#include "ops/dtype.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
ValuePtr DTypeInferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("dtype infer", int64_t(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  const std::set<TypePtr> valid_types = {kTensorType};
  auto type =
    CheckAndConvertUtils::CheckTensorTypeValid("infer type", input_args[0]->BuildType(), valid_types, op_name);
  return type;
}

AbstractBasePtr DTypeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto value = DTypeInferValue(primitive, input_args);
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(type);
  return abstract::MakeAbstract(std::make_shared<abstract::NoShape>(), type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(DType, prim::kPrimDType, DTypeInfer, DTypeInferValue, false);
}  // namespace ops
}  // namespace mindspore
