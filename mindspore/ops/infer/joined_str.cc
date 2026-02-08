/**
 * Copyright  2019-2023 Huawei Technologies Co., Ltd
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
#include "infer/joined_str.h"

#include <memory>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "primitive/structure_ops.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
bool ContainsVariableAbstractDict(const AbstractBasePtr &abstract) {
  if (abstract->isa<abstract::AbstractDictionary>()) {
    return abstract->BuildValue() == kValueAny;
  }
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  auto abstract_sequence = abstract->cast<abstract::AbstractSequencePtr>();
  const auto &elements = abstract_sequence->elements();
  return std::any_of(elements.begin(), elements.end(), [](const auto &e) { return ContainsVariableAbstractDict(e); });
}
}  // namespace

class JoinedStrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) const override {
    return std::make_shared<abstract::NoShape>();
  }

  TypePtr InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    for (const AbstractBasePtr &abstract : input_args) {
      if (ContainsVariableAbstractDict(abstract)) {
        MS_LOG(EXCEPTION) << "For JoinedStr, do not support dict input with variable elements, current abstract is: "
                          << abstract->ToString();
      }
    }
    return std::make_shared<String>();
  }

  ValuePtr InferValue(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    std::string res;
    for (const auto &arg : input_args) {
      if (ContainsVariableAbstractDict(arg)) {
        MS_LOG(EXCEPTION) << "For JoinedStr, do not support dict input with variable elements, current abstract is : "
                          << arg->ToString();
      }
      auto arg_value = arg->GetValue();
      MS_EXCEPTION_IF_NULL(arg_value);
      res += arg_value->ToString();
    }
    return MakeValue(res);
  }
};
MIND_API_OPERATOR_IMPL(JoinedStr, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(JoinedStr, prim::kPrimJoinedStr, JoinedStrInfer, true);
}  // namespace ops
}  // namespace mindspore
