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

#include "utils/infer_base.h"

#include <algorithm>
#include <memory>
#include <map>

#include "utils/check_convert_utils.h"

namespace mindspore {
abstract::ShapePtr InferBase::SingleInputOutputInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, primitive->name());
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr InferBase::SingleInputOutputInferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, primitive->name());
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  auto tensor_type = infer_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return data_type;
}

TypePtr InferBase::LogicalInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  const std::set<TypeId> valid_types = {kNumberTypeBool};
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("y", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (infer_type == kNumberTypeBool) {
    return TypeIdToType(infer_type);
  }
  return std::make_shared<TensorType>(TypeIdToType(kNumberTypeBool));
}

TypePtr InferBase::CheckSameInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                      const std::set<TypeId> valid_types, size_t input_num) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, SizeToLong(input_num),
                                     primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < input_num; i++) {
    types.emplace(std::to_string(i), input_args[i]->BuildType());
  }
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return TypeIdToType(infer_type);
}
}  // namespace mindspore
