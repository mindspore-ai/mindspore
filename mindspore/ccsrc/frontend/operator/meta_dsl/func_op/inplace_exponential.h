/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_INPLACE_EXPONENTIAL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_INPLACE_EXPONENTIAL_H_

#include <vector>
#include <memory>
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
void CheckInplaceExponentialInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args);
REGISTER_FUNCTION_OP(InplaceExponential, CheckInplaceExponentialInputs);
REGISTER_FUNCTION_OP(InplaceExponentialGrad);
PRIMITIVE_BPROP_REG(InplaceExponential, InplaceExponentialGrad);
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_INPLACE_EXPONENTIAL_H_
