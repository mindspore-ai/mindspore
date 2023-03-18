/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/make_range.h"

#include <vector>
#include <memory>
#include <string>
#include <set>

#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
bool CheckMakeRangeInput(const std::vector<AbstractBasePtr> &input_args, const std::string &prim_name) {
  constexpr size_t max_args_size = 3;
  constexpr size_t min_args_size = 1;
  auto inputs_size = input_args.size();
  if (inputs_size > max_args_size || inputs_size < min_args_size) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', the input size should within [" << min_args_size << ", "
                      << max_args_size << "] but got" << inputs_size;
  }
  bool has_variable = false;
  for (size_t i = 0; i < input_args.size(); ++i) {
    auto element = input_args[i];
    MS_EXCEPTION_IF_NULL(element);
    auto element_type = element->BuildType();
    if (element_type->type_id() != kInt64->type_id()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the " << i << "th input should be a int64 scalar but got "
                              << element->ToString();
    }
    if (!has_variable && element->BuildValue() == kValueAny) {
      has_variable = true;
    }
  }
  return has_variable;
}

abstract::AbstractTuplePtr CalcSlidePara(const std::vector<int64_t> &values, const std::string &prim_name) {
  auto values_size = values.size();
  int64_t start = values_size == 1 ? 0LL : values[kIndex0];
  int64_t stop = values_size == 1 ? values[kIndex0] : values[kIndex1];
  int64_t step = values_size <= kDim2 ? 1LL : values[kIndex2];

  if (step == 0) {
    MS_LOG(EXCEPTION) << "For 'range', the argument 'step' could not be 0.";
  }

  AbstractBasePtrList args;
  if (start <= stop) {
    if (step <= 0) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', when the argument 'start' " << start
                        << " is less than or equal to the argument 'stop' " << stop << ", "
                        << "the argument 'step' must be greater than 0, but the argument 'step' is " << step << ".";
    }

    for (int64_t i = start; i < stop; i += step) {
      args.push_back(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(i)));
      if (i > 0 && INT_MAX - i < step) {
        MS_EXCEPTION(ValueError) << "Integer overflow error occurred when traversing the range. "
                                 << "Please check the inputs of range.";
      }
    }
  } else {
    if (step >= 0) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', while the argument 'start' " << start
                        << " is greater than the argument "
                        << "'stop' " << stop << ", the argument 'step' must be less than 0, "
                        << "but the argument 'step' is " << step << ".";
    }

    for (int64_t i = start; i > stop; i += step) {
      args.push_back(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(i)));
      if (i < 0 && INT_MIN - i > step) {
        MS_EXCEPTION(ValueError) << "Integer overflow error occurred when traversing the range. "
                                 << "Please check the inputs of range.";
      }
    }
  }
  return std::make_shared<abstract::AbstractTuple>(args);
}

AbstractBasePtr InferImplMakeRange(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  bool has_variable = CheckMakeRangeInput(args_spec_list, prim_name);
  if (has_variable) {
    // If the input to make_range has variable input, the output abs should be dynamic length sequence.
    auto element = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
    auto ret = std::make_shared<abstract::AbstractTuple>(AbstractBasePtrList{element});
    ret->CheckAndConvertToDynamicLenSequence();
    return ret;
  }
  std::vector<int64_t> values;
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    auto element = args_spec_list[i];
    auto element_val = element->BuildValue();
    if (!element_val->isa<Int64Imm>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the " << i << "th input should be a int64 scalar but got "
                              << element->ToString();
    }
    values.push_back(element_val->cast<Int64ImmPtr>()->value());
  }
  return CalcSlidePara(values, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(make_range, BaseOperator);

// AG means auto generated
class MIND_API AGMakeRangeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InferImplMakeRange(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InferImplMakeRange(primitive, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InferImplMakeRange(primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0, 1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(make_range, prim::kPrimMakeRange, AGMakeRangeInfer, false);
}  // namespace ops
}  // namespace mindspore
