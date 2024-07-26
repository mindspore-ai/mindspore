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
#include <map>
#include <set>
#include <string>
#include "ops/ops_func_impl/cross.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
void CheckCrossDim(const std::string &prim_name, const ShapeVector &input_shape, const int64_t &dim_value) {
  const int64_t default_dim = -65530;
  const int64_t dim_size_value = 3;
  if (dim_value == default_dim) {
    auto iter = std::find(input_shape.begin(), input_shape.end(), dim_size_value);
    if (iter == input_shape.end()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the size of inputs dim must contain 3, but got "
                               << input_shape << ".";
    }
  } else {
    const auto shape_size = static_cast<int64_t>(input_shape.size());
    if (dim_value < -shape_size || dim_value >= shape_size) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', dim must be between "
                               << -static_cast<int64_t>(input_shape.size()) << " and "
                               << static_cast<int64_t>(input_shape.size()) - 1 << " , but got " << dim_value << ".";
    }
  }
}
}  // namespace

size_t CalCrossDimFromDefaultValue(const ShapeVector &input_shape, const ShapeVector &other_shape) {
  const int64_t dim_size_value = 3;
  auto broadcast_shape = CalBroadCastShape(input_shape, other_shape, "cross");
  for (size_t i = 0; i < broadcast_shape.size(); i++) {
    if (broadcast_shape[i] == dim_size_value) {
      return i;
    }
  }
  MS_EXCEPTION(ValueError) << "For cross, the size of inputs dim must contain 3, but got " << broadcast_shape << ".";
}

BaseShapePtr CrossFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto other_shape_ptr = input_args[kInputIndex1]->GetShape();
  // support dynamic rank
  if (!input_shape_ptr->isa<abstract::NoShape>() && !other_shape_ptr->isa<abstract::NoShape>()) {
    auto &input_shape = input_shape_ptr->GetShapeVector();
    auto &other_shape = other_shape_ptr->GetShapeVector();
    bool is_dynamic = IsDynamic(input_shape) || IsDynamic(other_shape);
    if (!is_dynamic) {
      auto broadcast_shape = CalBroadCastShape(input_shape, other_shape, primitive->name(), "input", "other");
      auto dim = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
      if (dim.has_value()) {
        CheckCrossDim(primitive->name(), broadcast_shape, dim.value());
      }
      return std::make_shared<abstract::TensorShape>(broadcast_shape);
    }
  }
  return input_shape_ptr->Clone();
}

TypePtr CrossFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto input_type = input_args[kIndex0]->GetType();
  return input_type;
}

TypePtrList CrossFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

ShapeArray CrossFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto &input_shape = input_tensor->shape();
  const auto &other_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(other_tensor);
  auto other_shape = other_tensor->shape();
  bool is_dynamic = IsDynamic(input_shape) || IsDynamic(other_shape);
  if (!is_dynamic) {
    auto broadcast_shape = CalBroadCastShape(input_shape, other_shape, primitive->name(), "input", "other");
    const auto &dim = input_values[kInputIndex2]->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(dim);
    auto dim_value = dim->value();
    CheckCrossDim(primitive->name(), broadcast_shape, dim_value);
    return {broadcast_shape};
  }
  return {input_shape};
}

REGISTER_SIMPLE_INFER(kNameCross, CrossFuncImpl)
}  // namespace ops
}  // namespace mindspore
