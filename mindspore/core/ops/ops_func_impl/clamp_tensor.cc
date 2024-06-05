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

#include "ops/ops_func_impl/clamp_tensor.h"
#include <vector>
#include <map>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
TypePtr ClampTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_type = input_args[kInputIndex0]->GetType();
  auto input1_type = input_args[kInputIndex1]->GetType();
  auto input2_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(input0_type);
  MS_EXCEPTION_IF_NULL(input1_type);
  MS_EXCEPTION_IF_NULL(input2_type);
  if (input1_type->type_id() == kMetaTypeNone && input2_type->type_id() == kMetaTypeNone) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  if (input0_type->type_id() == kNumberTypeBool || input1_type->type_id() == kNumberTypeBool ||
      input2_type->type_id() == kNumberTypeBool) {
    MS_EXCEPTION(ValueError) << "For Clamp, the dtype of 'input', 'min' and 'max' must not be bool.";
  }

  return input0_type->Clone();
}

TypePtrList ClampTensorFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  if (input_values[kInputIndex1] == mindspore::kNone && input_values[kInputIndex2] == mindspore::kNone) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }
  if (x_tensor->data_type() == kNumberTypeBool ||
      (input_values[kInputIndex1]->type() != nullptr &&
       input_values[kInputIndex1]->type()->type_id() == kNumberTypeBool) ||
      (input_values[kInputIndex2]->type() != nullptr &&
       input_values[kInputIndex2]->type()->type_id() == kNumberTypeBool)) {
    MS_EXCEPTION(ValueError) << "For Clamp, the dtype of 'input', 'min' and 'max' must not be bool.";
  }
  return {x_tensor->Dtype()};
}

bool ClampTensorFuncImpl::IsBroadcastable(const std::vector<int64_t> &x_shape,
                                          const std::vector<int64_t> &y_shape) const {
  if (x_shape == y_shape) {
    return true;
  }

  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return true;
  }

  if (x_shape.size() < y_shape.size()) {
    return false;
  }

  auto miss = x_shape.size() - y_shape.size();
  for (size_t i = 0; i < y_shape.size(); i++) {
    if (x_shape[miss + i] == y_shape[i]) {
      continue;
    }
    if (x_shape[miss + i] == -1) {
      continue;
    }
    if (y_shape[i] == -1 || y_shape[i] == 1) {
      continue;
    }
    return false;
  }
  return true;
}

BaseShapePtr ClampTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input0_shape);
  auto input1_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input1_shape);
  if (!input1_shape->isa<abstract::NoShape>()) {
    auto x_shape = input0_shape->GetShapeVector();
    auto min_shape = input1_shape->GetShapeVector();
    if (!IsBroadcastable(x_shape, min_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'min' "
                               << min_shape << " cannot broadcast.";
    }
  }
  auto input2_shape = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(input2_shape);
  if (!input2_shape->isa<abstract::NoShape>()) {
    auto x_shape = input0_shape->GetShapeVector();
    auto max_shape = input2_shape->GetShapeVector();
    if (!IsBroadcastable(x_shape, max_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'max' "
                               << max_shape << " cannot broadcast.";
    }
  }
  return input0_shape->Clone();
}

ShapeArray ClampTensorFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  if (input_values[kIndex1] == mindspore::kNone && input_values[kIndex2] == mindspore::kNone) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  const auto &input0 = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input0);
  auto x_shape = input0->shape();

  if (input_values[kIndex1] != mindspore::kNone) {
    const auto &input1 = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input1);
    auto min_shape = input1->shape();
    if (!IsBroadcastable(x_shape, min_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'min' "
                               << min_shape << " cannot broadcast.";
    }
  }

  if (input_values[kIndex2] != mindspore::kNone) {
    const auto &input2 = input_values[kIndex2]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input2);
    auto max_shape = input2->shape();
    if (!IsBroadcastable(x_shape, max_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'max' "
                               << max_shape << " cannot broadcast.";
    }
  }

  return {x_shape};
}
REGISTER_SIMPLE_INFER(kNameClampTensor, ClampTensorFuncImpl)
}  // namespace mindspore::ops
