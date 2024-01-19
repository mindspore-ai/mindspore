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

#include "ops/ops_func_impl/list_to_tuple.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ListToTupleFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  // Input list convert to tuple in backend
  // because UnifyIR use MakeTuple node in inplace of make_list node in backend_base.cc.
  auto input_shape_ptr = input_args[kIndex0]->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  auto input_shape = input_shape_ptr->shape();
  if (!input_shape.empty()) {
    auto first_len = input_shape[0];
    MS_EXCEPTION_IF_NULL(first_len);
    if (std::any_of(input_shape.begin() + 1, input_shape.end(), [first_len](BaseShapePtr len) {
          MS_EXCEPTION_IF_NULL(len);
          return *first_len != *len;
        })) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', each input of list should has same size in dynamic length case.";
    }
  }
  return std::make_shared<abstract::TupleShape>(input_shape);
}

TypePtr ListToTupleFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType()->cast<ListPtr>();
  MS_EXCEPTION_IF_NULL(input_type);
  return std::make_shared<List>(input_type->elements());
}
}  // namespace ops
}  // namespace mindspore
