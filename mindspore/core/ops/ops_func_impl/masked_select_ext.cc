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
#include "ops/ops_func_impl/masked_select_ext.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kMaskedSelectInputNum = 2;
TypePtr MaskedSelectExtInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[kIndex1]->GetType(), {kBool}, prim_name);
  return input_args[kIndex0]->GetType()->Clone();
}

BaseShapePtr MaskedSelectExtFrontendInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeDimAny}));
}
}  // namespace

BaseShapePtr MaskedSelectExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto mask_shape = input_args[kIndex1]->GetShape()->GetShapeVector();

  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  int64_t num = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int64_t>());
  ShapeVector real_shape = {num};
  return std::make_shared<abstract::TensorShape>(real_shape);
}

TypePtr MaskedSelectExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return MaskedSelectExtInferType(primitive, input_args);
}

class MaskedSelectExtFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto infer_type = MaskedSelectExtInferType(primitive, input_args);
    auto infer_shape = MaskedSelectExtFrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("MaskedSelectExt", MaskedSelectExtFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
