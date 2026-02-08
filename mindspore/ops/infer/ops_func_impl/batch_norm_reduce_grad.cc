/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/batch_norm_reduce_grad.h"
#include <memory>
#include "mindapi/base/types.h"
#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormReduceGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto weight_shape = input_args[kInputIndex4]->GetShape()->GetShapeVector();

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    std::make_shared<abstract::TensorShape>(weight_shape), std::make_shared<abstract::TensorShape>(weight_shape),
    std::make_shared<abstract::TensorShape>(weight_shape), std::make_shared<abstract::TensorShape>(weight_shape)});
}

TypePtr BatchNormReduceGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type_ptr = input_args[kInputIndex1]->GetType();
  auto weight_type_ptr = input_args[kInputIndex4]->GetType();
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{x_type_ptr->Clone(), x_type_ptr->Clone(), weight_type_ptr->Clone(), weight_type_ptr->Clone()});
}

}  // namespace ops
}  // namespace mindspore
