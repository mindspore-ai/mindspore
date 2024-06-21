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

#include "ops/ops_func_impl/constant_pad_nd.h"
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
BaseShapePtr ConstantPadNDFuncImpl::InferShape(const PrimitivePtr &primitive,
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
  if (x_rank == 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the dimension of 'x' must bigger than 0.";
  }
  // padding is dynamic
  ShapeVector out_shape;
  auto paddings_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (!paddings_opt.has_value()) {
    out_shape.resize(x_rank, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(std::move(out_shape));
  }
  constexpr size_t kScaleNum = 2;
  auto paddings = paddings_opt.value();
  if (!(paddings.size() % kScaleNum == 0)) {
    MS_EXCEPTION(ValueError) << "Length of padding must be even but got " << paddings.size();
  }
  if (!(x_rank >= paddings.size() / kScaleNum)) {
    MS_EXCEPTION(ValueError) << "Length of padding must be no more than 2 * dim of the input. "
                             << "Length of padding is: " << paddings.size() << "，while the input's dim is:" << x_rank;
  }
  auto l_diff = x_rank - (paddings.size() / kScaleNum);
  for (size_t i = 0; i < l_diff; ++i) {
    (void)out_shape.emplace_back(x_shape[i]);
  }
  for (size_t i = 0; i < paddings.size() / kScaleNum; ++i) {
    auto pad_idx = paddings.size() - ((i + 1) * 2);
    if (x_shape[l_diff + i] == abstract::Shape::kShapeDimAny || paddings.IsValueUnknown(pad_idx) ||
        paddings.IsValueUnknown(pad_idx + 1)) {
      (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    } else {
      auto new_dim = x_shape[l_diff + i] + paddings[pad_idx] + paddings[pad_idx + 1];
      (void)out_shape.emplace_back(new_dim);
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ConstantPadNDFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[kInputIndex0]->GetType(), {kTensorType}, prim_name);
}
}  // namespace ops
}  // namespace mindspore
