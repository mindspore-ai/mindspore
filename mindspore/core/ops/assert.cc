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

#include <map>
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "ops/assert.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Assert::Init(const int64_t summarize) { set_summarize(summarize); }

void Assert::set_summarize(const int64_t summarize) { this->AddAttr(kSummarize, MakeValue(summarize)); }

int64_t Assert::get_summarize() const {
  auto value_ptr = GetAttr(kSummarize);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr AssertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto Assert_prim = primitive->cast<PrimAssertPtr>();
  MS_EXCEPTION_IF_NULL(Assert_prim);
  auto op_name = Assert_prim->name();
  TypePtr condition;
  if (!(input_args[0]->BuildType()->type_id() == kObjectTypeTensorType)) {
    auto condition_value = GetValue<std::vector<bool>>(input_args[0]->BuildValue());
    CheckAndConvertUtils::CheckInteger("condition's rank", condition_value.size(), kLessEqual, 1, op_name);
    if (condition_value.size() == 1) {
      CheckAndConvertUtils::CheckInteger("condition[0]", condition_value[0], kEqual, 1, op_name);
    }
    condition = TypeIdToType(kNumberTypeBool);
  } else {
    auto condition_shape =
      CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), op_name);
    CheckAndConvertUtils::CheckInteger("condition's rank", condition_shape[0], kLessEqual, 1, op_name);
    if (condition_shape[0] == 1) {
      auto condition_value = reinterpret_cast<bool *>(input_args[0]->BuildValue()->cast<tensor::TensorPtr>()->data_c());
      MS_EXCEPTION_IF_NULL(condition_value);
      //      auto condition_value = GetValue<bool>(input_args[0]->BuildValue());
      CheckAndConvertUtils::CheckInteger("condition[0]", *condition_value, kEqual, 1, op_name);
    }
    condition = input_args[0]->BuildType();
  }
  std::vector<int64_t> output_shape = {1};
  std::set<TypeId> local_bool = {kNumberTypeBool};
  std::map<std::string, TypePtr> args = {{"condition", condition}};
  CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, local_bool, op_name);
  auto inputs_type = input_args[1]->BuildType()->cast<TuplePtr>()->elements();
  for (auto dtype : inputs_type) {
    std::set<TypePtr> template_types = {TypeIdToType(kObjectTypeTensorType)};
    CheckAndConvertUtils::CheckSubClass("input", dtype, template_types, op_name);
  }
  return std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeInt32), output_shape);
}
REGISTER_PRIMITIVE_C(kNameAssert, Assert);
}  // namespace ops
}  // namespace mindspore
