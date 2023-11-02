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

#ifndef MINDSPORE_CORE_OPS_OPS_FRONTEND_FUNC_IMPL_SCALAR_ARITHMETIC_H_
#define MINDSPORE_CORE_OPS_OPS_FRONTEND_FUNC_IMPL_SCALAR_ARITHMETIC_H_

#include <vector>
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace ops {
class ScalarArithmeticFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};

class ScalarAddFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarDivFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarEqFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarFloorDivFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarGeFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarGtFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarLeFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarLtFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarModFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarMulFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarPowFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};

class ScalarSubFrontendFuncImpl : public ScalarArithmeticFrontendFuncImpl {};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FRONTEND_FUNC_IMPL_SCALAR_ARITHMETIC_H_
