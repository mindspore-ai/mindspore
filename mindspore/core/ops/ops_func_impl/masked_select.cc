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
#include <functional>
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/masked_select.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kMaskedSelectInputNum = 2;
TypePtr MaskedSelectInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return input_args[kIndex0]->GetType()->Clone();
}

BaseShapePtr MaskedSelectFrontendInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMaskedSelectInputNum, op_name);
  return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeDimAny}));
}
}  // namespace

BaseShapePtr MaskedSelectFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto mask_shape_ptr = input_args[kInputIndex1]->GetShape();
  // support dynamic rank
  auto &input_shape = input_shape_ptr->GetShapeVector();
  auto &mask_shape = mask_shape_ptr->GetShapeVector();
  bool is_dynamic = IsDynamic(input_shape) || IsDynamic(mask_shape);
  if (!is_dynamic) {
    auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, primitive->name(), "input", "mask");
    int64_t num = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int64_t>());
    ShapeVector real_shape = {num};
    return std::make_shared<abstract::TensorShape>(real_shape);
  }
  return input_shape_ptr->Clone();
}

TypePtr MaskedSelectFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return MaskedSelectInferType(primitive, input_args);
}

TypePtrList MaskedSelectFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

ShapeArray MaskedSelectFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto &input_shape = input_tensor->shape();
  const auto &mask_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(mask_tensor);
  auto mask_shape = mask_tensor->shape();
  bool is_dynamic = IsDynamic(input_shape) || IsDynamic(mask_shape);
  if (!is_dynamic) {
    auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, primitive->name(), "input", "mask");
    int64_t num = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int64_t>());
    return {ShapeVector{num}};
  }
  return {input_shape};
}

class MaskedSelectFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto infer_type = MaskedSelectInferType(primitive, input_args);
    auto infer_shape = MaskedSelectFrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kNameMaskedSelect, MaskedSelectFrontendFuncImpl);
REGISTER_SIMPLE_INFER(kNameMaskedSelect, MaskedSelectFuncImpl)
}  // namespace ops
}  // namespace mindspore
