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

#include "ops/ops_func_impl/xlogy.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
std::vector<int64_t> XlogyFuncImpl::BroadcastShape(const std::vector<int64_t> &x_shape, const string x_name,
                                                   const std::vector<int64_t> &y_shape, const string y_name) const {
  if (x_shape == y_shape) {
    return x_shape;
  }

  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return {-2};
  }

  if (x_shape.size() < y_shape.size()) {
    return BroadcastShape(y_shape, y_name, x_shape, x_name);
  }

  std::vector<int64_t> res = x_shape;
  auto miss = x_shape.size() - y_shape.size();
  for (size_t i = 0; i < y_shape.size(); i++) {
    if (x_shape[miss + i] == y_shape[i] || x_shape[miss + i] == -1 || y_shape[i] == 1) {
      continue;
    }

    if (y_shape[i] == -1) {
      res[miss + i] = -1;
      continue;
    }
    if (x_shape[miss + i] == 1) {
      res[miss + i] = y_shape[i];
      continue;
    }
    MS_EXCEPTION(ValueError) << "For XLogy, the shape of " << x_name << " " << x_shape << " and the shape of " << y_name
                             << " " << y_shape << " cannot broadcast.";
  }
  return res;
}

BaseShapePtr XlogyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input0_shape);
  auto input1_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input1_shape);
  auto x_shape = input0_shape->GetShapeVector();
  auto other_shape = input1_shape->GetShapeVector();
  auto output_shape = BroadcastShape(x_shape, "input", other_shape, "other");
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr XlogyFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_type = input_args[kInputIndex0]->GetType();
  auto x2_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x1_type);
  MS_EXCEPTION_IF_NULL(x2_type);
  auto promote_type = PromoteType(x1_type, x2_type, primitive->name());
  if (IsIntegralType(promote_type, true)) {
    return {std::make_shared<TensorType>(kFloat32)};
  }
  return {std::make_shared<TensorType>(promote_type)};
}

ShapeArray XlogyFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();

  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);
  auto y_shape_vector = y_tensor->shape();

  auto output_shape = BroadcastShape(x_shape_vector, "input", y_shape_vector, "other");
  return {output_shape};
}

TypePtrList XlogyFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x1_type = x_tensor->Dtype();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);
  auto x2_type = y_tensor->Dtype();
  auto promote_type = PromoteType(x1_type, x2_type, primitive->name());
  if (IsIntegralType(promote_type, true)) {
    return {kFloat32};
  }
  return {promote_type};
}

REGISTER_SIMPLE_INFER(kNameXlogy, XlogyFuncImpl)
}  // namespace ops
}  // namespace mindspore
