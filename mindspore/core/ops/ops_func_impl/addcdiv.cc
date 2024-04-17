/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/addcdiv.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr AddcdivFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto output_shape = input_shape_ptr->GetShapeVector();
  std::vector<std::string> input_names = {"input", "tensor1", "tensor2", "value"};
  if (MS_UNLIKELY(input_args.size() != input_names.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the number of inputs should be "
                             << input_names.size() << ", but got " << input_args.size();
  }
  for (size_t i = 1; i < input_args.size(); ++i) {
    auto input_shape = input_args[i]->GetShape()->GetShapeVector();
    output_shape = CalBroadCastShape(output_shape, input_shape, primitive->name(), input_names[i - 1], input_names[i]);
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr AddcdivFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt64};
  const std::set<TypePtr> value_types = {kFloat16, kFloat32, kFloat64, kInt64, kInt32};
  auto input_type = input_args[kInputIndex0]->GetType();
  auto tensor1_type = input_args[kInputIndex1]->GetType();
  auto tensor2_type = input_args[kInputIndex2]->GetType();
  auto value_type = input_args[kInputIndex3]->GetType();
  const auto &op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("tensor1", tensor1_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("tensor2", tensor2_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("value", value_type, value_types, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_type);
  (void)types.emplace("tensor1", tensor1_type);
  (void)types.emplace("tensor2", tensor2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
