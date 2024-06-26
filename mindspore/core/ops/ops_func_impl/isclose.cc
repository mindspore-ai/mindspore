/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <string>
#include <memory>
#include "ops/ops_func_impl/isclose.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr IsCloseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    const int MAX = 0x7fffffff;
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
    auto other_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
    int64_t input_size = 1;
    int64_t other_size = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size *= input_shape[i];
    }
    for (size_t i = 0; i < other_shape.size(); i++) {
      other_size *= other_shape[i];
    }
    if (input_size > MAX) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the size of tensor 'input' must be less than [2147483648], but got: "
                               << input_size << ".";
    }
    if (other_size > MAX) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the size of tensor 'other' must be less than [2147483648], but got: "
                               << other_size << ".";
    }
  }
  return BroadCastInferShape(op_name, input_args);
}

TypePtr IsCloseFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kBool,  kFloat16, kFloat32, kFloat64, kInt8,
                                         kInt16, kInt32,   kInt64,   kUInt8,   kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[0]->GetType());
  (void)types.emplace("other", input_args[1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<TensorType>(kBool);
}
}  // namespace ops
}  // namespace mindspore
