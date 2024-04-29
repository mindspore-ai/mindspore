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
#include "ops/ops_func_impl/triu.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr TriuFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape_vec))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  const int64_t kMinShapeSize = 2;
  auto input_shape_rank = SizeToLong(input_shape_vec.size());
  MS_CHECK_VALUE(input_shape_rank >= kMinShapeSize,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of input", input_shape_rank, kGreaterEqual,
                                                             kMinShapeSize, primitive));
  return std::make_shared<abstract::Shape>(input_shape_vec);
}

TypePtr TriuFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[kInputIndex0]->GetType();
  if (!CheckAndConvertUtils::IsScalar(input_args[kInputIndex1])) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'diagonal' must be a scalar type, but got type: "
                            << input_args[kInputIndex1]->GetType()->ToString();
  }
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
