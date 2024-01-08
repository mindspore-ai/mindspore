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

#include "ops/ops_func_impl/transpose.h"

#include <vector>
#include <memory>
#include <set>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"

namespace mindspore::ops {
namespace {
size_t NormalizeDimIdx(int64_t idx, size_t rank) {
  auto new_idx = idx >= 0 ? idx : (idx + SizeToLong(rank));
  return LongToSize(new_idx);
}
}  // namespace
BaseShapePtr TransposeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto perm_res = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    if (MS_UNLIKELY(!perm_res.has_value())) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
    }
    auto perm = perm_res.value();
    return std::make_shared<abstract::TensorShape>(ShapeVector(perm.size(), abstract::TensorShape::kShapeDimAny));
  }

  auto x_rank = x_shape.size();
  if (MS_UNLIKELY(!perm_res.has_value())) {
    ShapeVector out_shape(x_rank, abstract::TensorShape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  ShapeVector out_shape;
  out_shape.reserve(x_rank);
  auto perm = perm_res.value();

  MS_CHECK_VALUE(perm.size() == x_rank,
                 CheckAndConvertUtils::FormatCommMsg("For '", primitive->name(),
                                                     "', size of perm should equal to rank of x, but got ", perm.size(),
                                                     " and ", x_rank, "!"));
  std::set<size_t> seen;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (MS_UNLIKELY(perm.IsValueUnknown(i))) {
      out_shape.push_back(abstract::TensorShape::kShapeDimAny);
      continue;
    }

    MS_CHECK_VALUE(-SizeToLong(x_rank) <= perm[i] && perm[i] < SizeToLong(x_rank),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("perm value", perm[i], kIncludeLeft,
                                                               {-SizeToLong(x_rank), SizeToLong(x_rank)}, primitive));
    auto dim = NormalizeDimIdx(perm[i], x_rank);
    MS_CHECK_VALUE(seen.count(dim) == 0,
                   CheckAndConvertUtils::FormatCommMsg(
                     "For '", primitive->name(), "', perms should all be unique dim, but", dim, " is not unique!"));
    seen.insert(dim);
    out_shape.push_back(x_shape[dim]);
  }

  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr TransposeFuncImpl::InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}
}  // namespace mindspore::ops
