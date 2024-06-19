
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

#include "ops/ops_func_impl/greater.h"

#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr GreaterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  return BroadCastInferShape(primitive->name(), input_args);
}
TypePtrList GreaterFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {kBool};
}
ShapeArray GreaterFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}
REGISTER_SIMPLE_INFER(kNameGreater, GreaterFuncImpl)
}  // namespace ops
}  // namespace mindspore
