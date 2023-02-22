/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/dropout.h"

#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
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

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Dropout, BaseOperator);
void Dropout::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void Dropout::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, api::MakeValue(keep_prob));
}

float Dropout::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}
AbstractBasePtr InferImplDropout(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<abstract::AbstractBasePtr> &args_spec_list) {
  auto op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = abstract::CheckArg<abstract::AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  auto output_shape =
    std::make_shared<abstract::AbstractTensor>(x->element(), std::make_shared<abstract::Shape>(shape));
  AbstractBasePtrList ret = {output_shape, output_shape};
  return std::make_shared<abstract::AbstractTuple>(ret);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Dropout, prim::kPrimDropout, InferImplDropout, nullptr, true);
}  // namespace ops
}  // namespace mindspore
