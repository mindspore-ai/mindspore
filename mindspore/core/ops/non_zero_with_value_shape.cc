/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/non_zero_with_value_shape.h"

#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
// NonZeroWithValueShape
MIND_API_OPERATOR_IMPL(NonZeroWithValueShape, BaseOperator);
AbstractBasePtr NonZeroWithValueShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  constexpr size_t input_num = 3;
  abstract::CheckArgsSize(op_name, input_args, input_num);
  abstract::AbstractTensorPtr x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);

  MS_EXCEPTION_IF_NULL(x);
  auto x_shape = x->shape();
  MS_EXCEPTION_IF_NULL(x_shape);
  ShapeVector y_shape;

  int64_t rank_base = SizeToLong(x_shape->shape().size());
  int64_t max_size = 0;
  if (std::any_of(x_shape->shape().begin(), x_shape->shape().end(), [](int64_t dim) { return dim < 0; })) {
    max_size = std::accumulate(x_shape->max_shape().begin(), x_shape->max_shape().end(), 1, std::multiplies<int64_t>());
  } else {
    max_size = std::accumulate(x_shape->shape().begin(), x_shape->shape().end(), 1, std::multiplies<int64_t>());
  }
  (void)y_shape.emplace_back(rank_base);
  // Indices of elements that are non-zero
  (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
  ShapeVector max_shape = {rank_base, max_size};

  auto value =
    std::make_shared<abstract::AbstractTensor>(x->element(), std::make_shared<abstract::Shape>(y_shape, max_shape));
  auto index =
    std::make_shared<abstract::AbstractTensor>(kInt32, std::make_shared<abstract::Shape>(y_shape, max_shape));
  AbstractBasePtrList result = {value, index};
  return std::make_shared<abstract::AbstractTuple>(result);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NonZeroWithValueShape, prim::kPrimNonZeroWithValueShape, NonZeroWithValueShapeInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
