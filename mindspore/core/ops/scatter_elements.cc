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

#include <string>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace {
constexpr size_t kScatterElementsArgSize = 3;
}  // namespace

namespace mindspore {
namespace ops {
void ScatterElements::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
int64_t ScatterElements::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

AbstractBasePtr ScatterElementsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const abstract::AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kScatterElementsArgSize);
  auto x = abstract::CheckArg<abstract::AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  return std::make_shared<abstract::AbstractTensor>(x->element(), std::make_shared<abstract::Shape>(shape));
}

MIND_API_OPERATOR_IMPL(ScatterElements, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterElements, prim::kPrimScatterElements, ScatterElementsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
