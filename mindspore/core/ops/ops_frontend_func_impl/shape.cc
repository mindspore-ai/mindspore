/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_frontend_func_impl.h"

namespace mindspore::ops {
class ShapeFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto in_shape_ptr = input_args[0]->GetShape();
    const auto &inshape = in_shape_ptr->GetShapeVector();
    if (IsDynamic(inshape)) {
      // If the input of shape is dynamic shape/rank tensor, value can not be directly built.
      // Run infer of shape.
      return nullptr;
    }
    return MakeValue(inshape);
  }

  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    auto input_shape_ptr = input_args[kIndex0]->GetShape();
    const auto &in_shape = input_shape_ptr->GetShapeVector();
    AbstractBasePtrList abs_list;
    // Set shape value to kValueAny when dynamic shape/rank.
    (void)std::transform(in_shape.begin(), in_shape.end(), std::back_inserter(abs_list),
                         [](int64_t item) -> std::shared_ptr<abstract::AbstractScalar> {
                           auto ret = std::make_shared<abstract::AbstractScalar>(item);
                           if (item == abstract::Shape::kShapeRankAny || item == abstract::Shape::kShapeDimAny) {
                             ret->set_value(kValueAny);
                           }
                           return ret;
                         });
    auto abs = std::make_shared<abstract::AbstractTuple>(abs_list);
    if (IsDynamicRank(in_shape)) {
      abs->CheckAndConvertToDynamicLenSequence();
    }
    return abs;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Shape", ShapeFrontendFuncImpl);
}  // namespace mindspore::ops
