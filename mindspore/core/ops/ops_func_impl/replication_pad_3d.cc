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

#include "ops/ops_func_impl/replication_pad_3d.h"
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReplicationPad3DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  const size_t kNum1 = 1;
  const size_t kNum2 = 2;
  const size_t kNum3 = 3;
  const size_t kRank4DNum = 4;
  const size_t kRank5DNum = 5;
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  // input x dynamic rank
  MS_EXCEPTION_IF_NULL(x_base_shape);
  if (x_base_shape->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  // input x dynamic shape
  auto x_rank = x_shape.size();
  if (x_rank != kRank4DNum && x_rank != kRank5DNum) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input should be 4D or 5D, but got " << x_rank;
  }
  // padding
  auto paddings_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (!paddings_opt.has_value()) {
    ShapeVector out_shape = x_shape;
    out_shape[x_rank - kNum1] = abstract::Shape::kShapeDimAny;
    out_shape[x_rank - kNum2] = abstract::Shape::kShapeDimAny;
    out_shape[x_rank - kNum3] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::Shape>(std::move(out_shape));
  }

  auto padding_type = input_args[kInputIndex1]->GetType();
  if (!padding_type->isa<Tuple>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', type of 'padding' should be tuple of int, but got"
                            << padding_type;
  }
  auto paddings = paddings_opt.value();
  const size_t kExpectedPaddingLength = 6;
  if (paddings.size() != kExpectedPaddingLength) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The padding length should be 6, but got "
                             << paddings.size();
  }
  auto out_shape = SetPadShape(x_shape, paddings);
  return out_shape;
}

TypePtr ReplicationPad3DFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[kInputIndex0]->GetType(), {kTensorType}, prim_name);
}
}  // namespace ops
}  // namespace mindspore
