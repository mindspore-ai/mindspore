/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/static_analysis/param_validator.h"
#include "pipeline/static_analysis/prim.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/utils.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplScalarSummary(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a scalar and a tensor or scalar.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);

  // check the tag
  AbstractScalarPtr descriptions = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);

  // check the value: scalar or shape = (1,)
  auto scalar_value = dyn_cast<AbstractScalar>(args_spec_list[1]);
  if (scalar_value == nullptr) {
    auto tensor_value = dyn_cast<AbstractTensor>(args_spec_list[1]);
    if (tensor_value == nullptr) {
      MS_LOG(EXCEPTION) << "Input must be scalar or shape(1,)";
    }
  } else {
    auto item_v = scalar_value->BuildValue();
    if (item_v->isa<StringImm>()) {
      auto value = item_v->cast<StringImmPtr>()->value();
      if (value.empty()) {
        MS_LOG(EXCEPTION) << "Input summary value can't be null";
      }
    }
  }

  // Reomve the force check to support batch set summary use 'for' loop
  auto item_v = descriptions->BuildValue();
  if (!item_v->isa<StringImm>()) {
    MS_EXCEPTION(TypeError) << "Summary first parameter should be string";
  }

  return std::make_shared<AbstractScalar>(kAnyValue, kBool);
}

AbstractBasePtr InferImplTensorSummary(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a scalar(tag) and a tensor(value)
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);

  // check the tag
  auto descriptions = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  auto tensor_value = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);

  int tensor_rank = SizeToInt(tensor_value->shape()->shape().size());
  if (tensor_rank == 0) {
    MS_LOG(EXCEPTION) << op_name << " summary evaluator second arg should be an tensor, but got a scalar, rank is 0";
  }

  // Reomve the force check to support batch set summary use 'for' loop
  auto item_v = descriptions->BuildValue();
  if (!item_v->isa<StringImm>()) {
    MS_EXCEPTION(TypeError) << "Summary first parameter should be string";
  }

  return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<Bool>());
}
}  // namespace abstract
}  // namespace mindspore
