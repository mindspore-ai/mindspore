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

#include "ops/is_close.h"
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
namespace {
abstract::ShapePtr IsCloseInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const int MAX = 0x3fffffff;
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto other_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto input_rank = SizeToLong(input_shape.size());
  auto other_rank = SizeToLong(other_shape.size());
  CheckAndConvertUtils::Check("input rank", input_rank, kEqual, other_rank, op_name);
  int64_t input_size = 1, other_size = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size *= input_shape[i];
    other_size *= other_shape[i];
    if (input_shape[i] != other_shape[i] && (input_shape[i] != 1 || other_shape[i] != 1)) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', The size of tensor input must match the size of tensor other at the " << i
                               << " dimension!";
    }
  }
  if (input_size > MAX)
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', The size of tensor input must should be less than [2147483648], actual is "
                             << input_size;
  if (other_size > MAX)
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', The size of tensor other must should be less than [2147483648], actual is "
                             << other_size;
  return BroadCastInferShape(op_name, input_args);
}

TypePtr IsCloseInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt32};
  std::map<std::string, TypePtr> types;
  types.emplace("input", input_args[0]->BuildType());
  types.emplace("other", input_args[1]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(), valid_types, op_name);
  CheckAndConvertUtils::CheckTensorTypeValid("other", input_args[1]->BuildType(), valid_types, op_name);
  CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<TensorType>(kBool);
}
}  // namespace
AbstractBasePtr IsCloseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = IsCloseInferType(primitive, input_args);
  auto infershape = IsCloseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(IsClose, prim::kPrimIsClose, IsCloseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
