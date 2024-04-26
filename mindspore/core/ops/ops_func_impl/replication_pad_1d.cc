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

#include "ops/ops_func_impl/replication_pad_1d.h"
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
BaseShapePtr ReplicationPad1DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
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
  if (x_rank != 2 && x_rank != 3) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input should be 2D or 3D, but got " << x_rank;
  }
  // padding
  auto paddings_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (!paddings_opt.has_value()) {
    ShapeVector out_shape = x_shape;
    out_shape[x_rank - 1] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::Shape>(std::move(out_shape));
  }

  auto padding_type = input_args[kInputIndex1]->GetType();
  if (!padding_type->isa<Tuple>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', type of 'padding' should be tuple of int, but got"
                            << padding_type;
  }
  auto paddings = paddings_opt.value();
  if (paddings.size() != 2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The padding length should be 2, but got "
                             << paddings.size();
  }
  auto out_shape = SetPadShape(x_shape, paddings);
  return out_shape;
}

TypePtr ReplicationPad1DFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[kInputIndex0]->GetType(), {kTensorType}, prim_name);
}
}  // namespace ops
}  // namespace mindspore
