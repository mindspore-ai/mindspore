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

#include "ops/square_sum_v1.h"

#include <memory>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr SquareSumV1InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return ReduceBaseInferShape(primitive, input_args, "square_sum_v1");
}

TypePtr SquareSumV1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types_with_complex, prim->name());
  return x_type;
}

AbstractBasePtr SquareSumV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = SquareSumV1InferType(primitive, input_args);
  auto shapes = ReduceBaseInferShape(primitive, input_args, "square_sum_v1");
  return abstract::MakeAbstract(shapes, types);
}
}  // namespace

void SquareSumV1::Init(int axis, bool keep_dims) {
  this->set_axis(axis);
  this->set_keep_dims(keep_dims);
}

void SquareSumV1::set_keep_dims(bool keep_dims) { (void)this->AddAttr(kNameKeepDims, api::MakeValue(keep_dims)); }

bool SquareSumV1::get_keep_dims() const { return GetValue<bool>(GetAttr(kNameKeepDims)); }

void SquareSumV1::set_axis(int64_t axis) { (void)this->AddAttr(kNameAxis, api::MakeValue(axis)); }

int64_t SquareSumV1::get_axis() const { return GetValue<int64_t>(GetAttr(kNameAxis)); }

MIND_API_OPERATOR_IMPL(SquareSumV1, BaseOperator);
class MIND_API AGSquareSumV1Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumV1InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumV1InferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumV1Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SquareSumV1, prim::kSquareSumV1, AGSquareSumV1Infer, false);
}  // namespace ops
}  // namespace mindspore
