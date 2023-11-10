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

#include "ops/ops_func_impl/geqrf.h"

#include <algorithm>
#include <string>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr GeqrfFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  BaseShapePtr base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &a_shape = base_shape->GetShapeVector();

  if (IsDynamicRank(a_shape)) {
    ShapeVector dyn_shape{abstract::TensorShape::kShapeRankAny};
    std::vector<abstract::BaseShapePtr> shape_tuple;
    (void)shape_tuple.emplace_back(std::make_shared<abstract::TensorShape>(dyn_shape));
    (void)shape_tuple.emplace_back(std::make_shared<abstract::TensorShape>(dyn_shape));
    return std::make_shared<abstract::TupleShape>(shape_tuple);
  }
  auto ndim = a_shape.size();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const size_t kTwo = 2;
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    MS_CHECK_VALUE(ndim == kTwo,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of a", SizeToLong(ndim), kEqual, kTwo, primitive));
  } else {
    MS_CHECK_VALUE(ndim >= kTwo, CheckAndConvertUtils::FormatCheckIntegerMsg("rank of a", SizeToLong(ndim),
                                                                             kGreaterEqual, kTwo, primitive));
  }
  auto m = a_shape[ndim - 2];
  auto n = a_shape[ndim - 1];
  int64_t p = std::min(m, n);
  std::vector<int64_t> tau_shape;
  auto offset = ndim - static_cast<size_t>(kDim2);
  for (size_t i = 0; i < offset; i++) {
    (void)tau_shape.emplace_back(a_shape[i]);
  }
  (void)tau_shape.emplace_back(p);

  std::vector<abstract::BaseShapePtr> shape_tuple;
  (void)shape_tuple.emplace_back(std::make_shared<abstract::TensorShape>(a_shape));
  (void)shape_tuple.emplace_back(std::make_shared<abstract::TensorShape>(tau_shape));
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TypePtr GeqrfFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(type);

  std::vector<TypePtr> type_tuple = {type, type};
  return std::make_shared<Tuple>(type_tuple);
}

}  // namespace ops
}  // namespace mindspore
