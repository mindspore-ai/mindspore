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

#include "ops/ops_func_impl/scalar_arithmetic.h"
#include <set>
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScalarArithmeticFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return abstract::kNoShape;
}

TypePtr ScalarArithmeticFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_type = input_args[kIndex0]->GetType();
  auto y_type = input_args[kIndex1]->GetType();
  std::set<std::string> compare_ops = {kNameScalarEq, kNameScalarGe, kNameScalarGt, kNameScalarLt, kNameScalarLe};
  auto iter = compare_ops.find(prim_name);
  if (iter != compare_ops.end()) {
    return kBool;
  }
  return HighPriorityType(x_type, y_type, prim_name);
}
}  // namespace ops
}  // namespace mindspore
