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

#include "ops/ops_func_impl/floor_mod.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FloorModFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input0_shape = input_args[kIndex0]->GetShape();
  auto input1_shape = input_args[kIndex1]->GetShape();

  const int64_t max_dim = 8;
  MS_CHECK_VALUE(input0_shape->GetShapeVector().size() < max_dim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("The dimension of FloorMod input",
                                                             SizeToLong(input0_shape->GetShapeVector().size()),
                                                             kLessThan, max_dim, primitive));
  MS_CHECK_VALUE(input1_shape->GetShapeVector().size() < max_dim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("The dimension of FloorMod input",
                                                             SizeToLong(input1_shape->GetShapeVector().size()),
                                                             kLessThan, max_dim, primitive));
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr FloorModFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto type_x = input_args[kIndex0]->GetType();
  auto type_y = input_args[kIndex1]->GetType();

  if (type_x->isa<Complex>() || type_y->isa<Complex>()) {
    if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeComplex64) {
      return type_x->Clone();
    } else if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeFloat32) {
      return type_x->Clone();
    } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeComplex128) {
      return type_x->Clone();
    } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeFloat64) {
      return type_x->Clone();
    } else if (type_x->type_id() == kNumberTypeFloat32 && type_y->type_id() == kNumberTypeComplex64) {
      return type_y->Clone();
    } else if (type_x->type_id() == kNumberTypeFloat64 && type_y->type_id() == kNumberTypeComplex128) {
      return type_y->Clone();
    } else {
      MS_EXCEPTION(TypeError)
        << "For '" << op_name
        << "', complex math binary op expecting Tensor [complex64, complex64], [complex64, float32], [float32, "
           "complex64], [complex128, complex128], [complex128, float64], [float64, complex128], but got ["
        << type_x->ToString() << ", " << type_y->ToString() << "].";
    }
  }

  return type_x->Clone();
}
}  // namespace ops
}  // namespace mindspore
