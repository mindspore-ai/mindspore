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

#include "infer/ops_func_impl/transpose_ext_view.h"

#include <vector>
#include <memory>
#include <set>
#include <utility>
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
BaseShapePtr TransposeExtViewFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto x_rank = SizeToLong(x_shape.size());
  auto dim0_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto dim1_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (!dim0_opt.has_value() || !dim1_opt.has_value()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{x_rank, abstract::Shape::kShapeDimAny});
  }
  auto dim0 = dim0_opt.value();
  auto dim1 = dim1_opt.value();
  // if x is a scalar tensor, then dim must be in the range [-1, 0].
  if (x_rank <= 0) {
    x_rank = 1;
  }
  MS_CHECK_VALUE(dim0 >= -x_rank && dim0 < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim0", dim0, kIncludeLeft, {-x_rank, x_rank}, primitive));
  MS_CHECK_VALUE(dim1 >= -x_rank && dim1 < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim1", dim1, kIncludeLeft, {-x_rank, x_rank}, primitive));
  auto tmp_dim0 = LongToSize((dim0 < 0) ? dim0 + x_rank : dim0);
  auto tmp_dim1 = LongToSize((dim1 < 0) ? dim1 + x_rank : dim1);
  if (tmp_dim0 == tmp_dim1) {
    return std::make_shared<abstract::TensorShape>(x_shape);
  }
  ShapeVector ret_shape = x_shape;
  std::swap(ret_shape[tmp_dim0], ret_shape[tmp_dim1]);
  return std::make_shared<abstract::TensorShape>(ret_shape);
}

TypePtr TransposeExtViewFuncImpl::InferType(const PrimitivePtr &,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType();
}

}  // namespace mindspore::ops
