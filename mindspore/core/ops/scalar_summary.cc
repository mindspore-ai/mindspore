/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "ops/scalar_summary.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {

MIND_API_OPERATOR_IMPL(ScalarSummary, BaseOperator);
void ScalarSummary::set_side_effect_io() { (void)this->AddAttr(kSideEffectIO, api::MakeValue(true)); }

bool ScalarSummary::get_side_effect_io() const {
  auto value_ptr = GetAttr(kSideEffectIO);
  return GetValue<bool>(value_ptr);
}

void ScalarSummary::Init() { this->set_side_effect_io(); }

class MIND_API ScalarSummaryInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    // check
    auto v_shape = input_args[1]->GetShape()->GetShapeVector();
    (void)CheckAndConvertUtils::CheckInteger("v rank", int64_t(v_shape.size()), kLessEqual, 1, prim_name);
    return std::make_shared<abstract::Shape>(ShapeVector(1));
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // check
    CheckAndConvertUtils::CheckSummaryParam(input_args[0], input_args[1], primitive->name());
    return kInt32;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarSummary, prim::kPrimScalarSummary, ScalarSummaryInfer, false);
}  // namespace ops
}  // namespace mindspore
