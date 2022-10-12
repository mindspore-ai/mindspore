/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/relu6.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReLU6, BaseOperator);
class ReLU6Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", int64_t(input_args.size()), kGreaterEqual, 1, prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    return input_args[0]->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
      MS_LOG(EXCEPTION) << "For '" << prim->name()
                        << ", the input args used for infer shape and type is necessary, but missing it.";
    }
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[0]->BuildType());
    return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReLU6, prim::kPrimReLU6, ReLU6Infer, false);
}  // namespace ops
}  // namespace mindspore
