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

#include "infer/ops_func_impl/bincount_ext.h"
#include <vector>
#include <set>
#include <memory>
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/ops_frontend_func_impl.h"
#include "utils/check_convert_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr BincountExtInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  // if weight is not None, check weight type
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    if (input_args[kInputIndex1]->GetType()->cast<TensorTypePtr>()->element()->type_id() == kNumberTypeFloat32) {
      return std::make_shared<TensorType>(kFloat32);
    }
    return std::make_shared<TensorType>(kFloat64);
  }

  return std::make_shared<TensorType>(kInt64);
}

BaseShapePtr BincountExtFrontendInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  // return input shape first and reshape in backend
  return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
}
}  // namespace

BaseShapePtr BincountExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  // support dynamic shape
  if (IsDynamic(input_shape_ptr->GetShapeVector())) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
  }
  // return input shape first and reshape in backend
  return input_shape_ptr;
}

TypePtr BincountExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return BincountExtInferType(primitive, input_args);
}

ShapeArray BincountExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  auto input_shape = input_tensor->shape();
  return {input_shape};
}

TypePtrList BincountExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  // check if weight tensor is exist
  if (input_values[kInputIndex1] != mindspore::kNone) {
    const auto &weight_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
    if (weight_tensor->Dtype()->type_id() == kNumberTypeFloat32) {
      return {kFloat32};
    }
    return {kFloat64};
  }

  return {kInt64};
}

class OPS_API BincountExtFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    auto infer_type = BincountExtInferType(primitive, input_args);
    auto infer_shape = BincountExtFrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kNameBincountExt, BincountExtFrontendFuncImpl);
REGISTER_SIMPLE_INFER(kNameBincountExt, BincountExtFuncImpl);
}  // namespace ops
}  // namespace mindspore
