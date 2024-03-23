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
#include "ops/ops_func_impl/batch_norm_grad_ext.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  std::vector<BaseShapePtr> shapes_list{x_shape_ptr->Clone(), weight_shape_ptr->Clone(), weight_shape_ptr->Clone()};
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr BatchNormGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto dy_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(dy_type);
  std::vector<TypePtr> types_list;
  types_list = {dy_type, kFloat32, kFloat32};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
