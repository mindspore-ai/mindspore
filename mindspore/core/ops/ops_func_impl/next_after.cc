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

#include "ops/ops_func_impl/next_after.h"
#include <map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
BaseShapePtr NextAfterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr NextAfterFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_infer_type = input_args[0]->GetType();
  auto x2_infer_type = input_args[1]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", x1_infer_type);
  (void)types.emplace("x2", x2_infer_type);
  const std::set<TypePtr> input_valid_types = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, input_valid_types, prim->name());
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
