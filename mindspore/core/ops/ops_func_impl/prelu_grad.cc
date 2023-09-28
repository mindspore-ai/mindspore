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

#include <vector>
#include <memory>
#include "abstract/dshape.h"
#include "utils/log_adapter.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/prelu_grad.h"

namespace mindspore::ops {
BaseShapePtr PReLUGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex1]->GetShape();
  auto w_shape = input_args[kInputIndex2]->GetShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, w_shape});
}

TypePtr PReLUGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex1]->GetType();
  auto w_type = input_args[kInputIndex2]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, w_type});
}

int32_t PReLUGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto dy_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto dy_shape = dy_shape_ptr->GetShapeVector();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto weight_shape = weight_shape_ptr->GetShapeVector();

  if (IsDynamic(dy_shape) || IsDynamic(x_shape) || IsDynamic(weight_shape)) {
    return OP_CHECK_RETRY;
  }

  if (dy_shape != x_shape) {
    MS_EXCEPTION(ShapeError) << "For '" << prim_name << "', the shape of 'y' and 'x' are not same"
                             << ", dy_shape: " << dy_shape_ptr->ToString() << ", x_shape: " << x_shape_ptr->ToString()
                             << ".";
  }

  auto weight_rank = weight_shape.size();
  MS_CHECK_VALUE(weight_rank == 1, "The dimension of 'weight' must be 1 .");

  auto x_rank = x_shape.size();
  auto channel_num = x_rank <= 1 ? 1 : x_shape[1];
  auto weight_len = weight_shape[0];
  if (weight_len != channel_num) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the length of 'weight' must be equal to number of channels: " << channel_num
                             << ", but got " << weight_shape << ".";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace mindspore::ops
