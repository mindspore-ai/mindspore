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

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SquareSumV1::Init(int axis, bool keep_dims) {
  this->set_axis(axis);
  this->set_keep_dims(keep_dims);
}

void SquareSumV1::set_keep_dims(bool keep_dims) { (void)this->AddAttr(kNameKeepDims, api::MakeValue(keep_dims)); }

bool SquareSumV1::get_keep_dims() const { return GetValue<bool>(GetAttr(kNameKeepDims)); }

void SquareSumV1::set_axis(int64_t axis) { (void)this->AddAttr(kNameAxis, api::MakeValue(axis)); }

int64_t SquareSumV1::get_axis() const { return GetValue<int64_t>(GetAttr(kNameAxis)); }

namespace {
TypePtr SquareSumV1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types_with_complex, prim->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SquareSumV1, BaseOperator);
AbstractBasePtr SquareSumV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = SquareSumV1InferType(primitive, input_args);
  auto shapes = ReduceBaseInferShape(primitive, input_args, "square_sum_v1");
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SquareSumV1, prim::kSquareSumV1, SquareSumV1Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
