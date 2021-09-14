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

void Assert::set_summarize(const int64_t summarize) { (void)this->AddAttr(kSummarize, MakeValue(summarize)); }

int64_t Assert::get_summarize() const {
  auto value_ptr = GetAttr(kSummarize);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr AssertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  TypePtr condition;
  if (!(input_args[0]->BuildType()->type_id() == kObjectTypeTensorType)) {
    auto condition_values = GetValue<std::vector<bool>>(input_args[0]->BuildValue());
    (void)CheckAndConvertUtils::CheckInteger("condition's rank", SizeToLong(condition_values.size()), kLessEqual, 1,
                                             op_name);
    if (condition_values.size() == 1) {
      if (!condition_values[0]) {
        MS_EXCEPTION(ValueError) << "condition value must be `true` when only one value contained.";
      }
    }
    condition = TypeIdToType(kNumberTypeBool);
  } else {
    auto condition_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("condition's rank", condition_shape[0], kLessEqual, 1, op_name);
    if (condition_shape[0] == 1) {
      auto condition_value = reinterpret_cast<bool *>(input_args[0]->BuildValue()->cast<tensor::TensorPtr>()->data_c());
      MS_EXCEPTION_IF_NULL(condition_value);
      if (!*condition_value) {
        MS_EXCEPTION(ValueError) << "condition value must be `true` when only one value contained.";
      }
    }
    condition = input_args[0]->BuildType();
  }
  std::vector<int64_t> output_shape = {1};
  std::set<TypePtr> local_bool = {kBool};
  std::map<std::string, TypePtr> args = {{"condition", condition}};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, local_bool, op_name);
  auto inputs_type = input_args[1]->BuildType()->cast<TuplePtr>()->elements();
  for (auto dtype : inputs_type) {
    std::set<TypePtr> template_types = {kTensorType};
    (void)CheckAndConvertUtils::CheckSubClass("input", dtype, template_types, op_name);
  }
  return std::make_shared<abstract::AbstractTensor>(kInt32, output_shape);
}
REGISTER_PRIMITIVE_C(kNameAssert, Assert);
}  // namespace ops
}  // namespace mindspore
