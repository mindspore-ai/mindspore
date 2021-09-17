/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <vector>
#include <memory>
#include <string>

#include "ops/assign.h"
#include "ops/op_utils.h"
#include "ir/dtype/ref.h"

namespace mindspore {
namespace ops {
AbstractBasePtr InferImplAssign(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger(
    "infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(args_spec_list)), kEqual, input_num, prim_name);
  auto check_types = common_valid_types;
  (void)check_types.emplace(kBool);
  auto variable_type = args_spec_list[0]->BuildType();
  auto value_type = args_spec_list[1]->BuildType();
  CheckAndConvertUtils::CheckScalarOrTensorTypesSame(std::map<std::string, TypePtr>{{"value", value_type}}, check_types,
                                                     prim_name);
  if (variable_type->isa<RefKeyType>()) {
    return args_spec_list[1]->Broaden();
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("variable", variable_type, check_types, prim_name);

  auto variable_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(args_spec_list[0]->BuildShape())[kShape];
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(args_spec_list[1]->BuildShape())[kShape];
  if (variable_shape.size() != value_shape.size()) {
    if (variable_shape.size() == 1 && variable_shape[0] == 1 && value_shape.empty()) {
      return args_spec_list[0];
    } else if (value_shape.size() == 1 && value_shape[0] == 1 && variable_shape.empty()) {
      return args_spec_list[0];
    } else {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the rank of value is " << value_shape.size()
                               << ". It should be same with variable's rank " << variable_shape.size() << ".";
    }
  }
  for (uint64_t i = 0; i < variable_shape.size(); i++) {
    if (variable_shape[i] != value_shape[i]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the shape of value is "
                               << args_spec_list[1]->BuildShape()->ToString()
                               << ". It should be same with variable's shape "
                               << args_spec_list[0]->BuildShape()->ToString() << ".";
    }
  }
  return args_spec_list[0];
}
REGISTER_PRIMITIVE_EVAL_IMPL(Assign, prim::kPrimAssign, InferImplAssign, nullptr, true);
}  // namespace ops
}  // namespace mindspore
