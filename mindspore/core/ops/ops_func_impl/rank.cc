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

#include <vector>
#include "ops/op_name.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/rank.h"

namespace mindspore::ops {
BaseShapePtr RankFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return abstract::kNoShape;
}

TypePtr RankFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  TypePtr res = kInt64;
  return res;
}

class RankFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(input_shape_vec)) {
      return kValueAny;
    }
    auto x_shape_rank = SizeToLong(input_shape_vec.size());
    ValuePtr res = MakeValue(x_shape_rank);
    return res;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Rank", RankFrontendFuncImpl);
}  // namespace mindspore::ops
