/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/inner_unique.h"
#include <memory>
#include <numeric>
#include <functional>
#include "ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray InnerUniqueFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto indice_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
  auto return_inverse_opt = input_infos[kIndex2]->GetScalarValue<bool>();
  if (MS_LIKELY(return_inverse_opt.has_value())) {
    indice_shape = return_inverse_opt.value() ? input_shape : ShapeVector{0};
  }

  auto output_shape = ShapeVector{abstract::TensorShape::kShapeDimAny};
  if (MS_LIKELY(!input_infos[kIndex0]->IsDynamic())) {
    output_shape = ShapeVector{
      std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>())};
  }

  return {output_shape, indice_shape};
}

std::vector<TypeId> InnerUniqueFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex0]->GetType();
  return {input_type, kNumberTypeInt64};
}

class OPS_API InnerUniqueFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_baseshape = input_args[kIndex0]->GetShape();
    MS_EXCEPTION_IF_NULL(input_baseshape);
    auto input_shape = input_baseshape->GetShapeVector();

    auto return_inverse_opt = GetScalarValue<bool>(input_args[kIndex2]->GetValue());
    auto indice_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
    if (MS_LIKELY(return_inverse_opt.has_value())) {
      indice_shape = return_inverse_opt.value() ? input_shape : ShapeVector{0};
    }
    auto output_shape = ShapeVector{abstract::TensorShape::kShapeDimAny};
    auto output_baseshape = std::make_shared<abstract::TensorShape>(output_shape);
    auto indice_baseshape = std::make_shared<abstract::TensorShape>(indice_shape);

    auto output_dtype = input_args[kIndex0]->GetType();
    MS_EXCEPTION_IF_NULL(output_dtype);

    auto output_tensor = abstract::MakeAbstractTensor(output_baseshape, output_dtype);
    auto indice_tensor = abstract::MakeAbstractTensor(indice_baseshape, kInt64);

    return std::make_shared<abstract::AbstractTuple>(AbstractBasePtrList{output_tensor, indice_tensor});
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("InnerUnique", InnerUniqueFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
