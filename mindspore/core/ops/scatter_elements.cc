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

#include "ops/scatter_elements.h"
#include <memory>
#include <string>
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr ScatterElementsInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  return input_args[0]->BuildShape();
}

TypePtr ScatterElementsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = primitive->name();
  constexpr int64_t kScatterElementsArgSize = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kScatterElementsArgSize, op_name);
  return input_args[0]->BuildType();
}

AbstractBasePtr ScatterElementsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(ScatterElementsInferShape(primitive, input_args),
                                ScatterElementsInferType(primitive, input_args));
}
}  // namespace

void ScatterElements::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
int64_t ScatterElements::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

MIND_API_OPERATOR_IMPL(ScatterElements, BaseOperator);
class MIND_API AGScatterElementsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterElementsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterElementsInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterElementsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterElements, prim::kPrimScatterElements, AGScatterElementsInfer, false);
}  // namespace ops
}  // namespace mindspore
