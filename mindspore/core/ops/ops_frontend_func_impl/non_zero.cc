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

#include "ops/ops_func_impl/non_zero.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {

class NonZeroFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    const auto &x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto x_rank = SizeToLong(x_shape.size());
    int64_t kNonZeroInputMinDim = 1;
    MS_CHECK_VALUE(x_rank >= kNonZeroInputMinDim,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("dimension of 'x'", x_rank, kGreaterEqual,
                                                               kNonZeroInputMinDim, primitive));
    if (IsDynamicRank(x_shape)) {
      x_rank = abstract::Shape::kShapeDimAny;
    }
    auto output_shape = ShapeVector({abstract::Shape::kShapeDimAny, x_rank});
    return std::make_shared<abstract::AbstractTensor>(kInt64, output_shape);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("NonZero", NonZeroFrontendFuncImpl);
}  // namespace mindspore::ops
