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

#include "infer/ops_func_impl/batch_norm_stats.h"
#include <set>
#include <vector>
#include <memory>
#include <utility>
#include "primitive/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kInputShapeMinShapeBNS = 2;
BaseShapePtr BatchNormStatsFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  ShapeVector final_mean_shape;
  ShapeVector final_invstd_shape;
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_mean_shape{abstract::TensorShape::kShapeDimAny};
    ShapeVector dyn_invstd_shape{abstract::TensorShape::kShapeDimAny};
    final_mean_shape = dyn_mean_shape;
    final_invstd_shape = dyn_invstd_shape;
  } else {
    MS_CHECK_VALUE(input_shape.size() >= kInputShapeMinShapeBNS,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("input shape", SizeToLong(input_shape.size()),
                                                               kGreaterEqual, kInputShapeMinShapeBNS, primitive));
    ShapeVector mean_shape{input_shape[kIndex1]};
    ShapeVector invstd_shape{input_shape[kIndex1]};
    final_mean_shape = mean_shape;
    final_invstd_shape = invstd_shape;
  }
  auto final_mean_shape_ptr = std::make_shared<abstract::TensorShape>(std::move(final_mean_shape));
  auto final_invstd_shape_ptr = std::make_shared<abstract::TensorShape>(std::move(final_invstd_shape));
  std::vector<abstract::BaseShapePtr> shape_tuple{final_mean_shape_ptr, final_invstd_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TypePtr BatchNormStatsFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  const std::set<TypePtr> input_valid_types = {kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_args[kInputIndex0]->GetType(), input_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kFloat32, kFloat32});
}
}  // namespace ops
}  // namespace mindspore
