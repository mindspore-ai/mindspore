/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/clip.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/lite_ops.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Clip, BaseOperator);
void Clip::Init(const float max, const float min) {
  this->set_max(max);
  this->set_min(min);
}

void Clip::set_max(const float max) { (void)this->AddAttr(kMax, api::MakeValue(max)); }

float Clip::get_max() const {
  auto value_ptr = this->GetAttr(kMax);
  return GetValue<float>(value_ptr);
}

void Clip::set_min(const float min) { (void)this->AddAttr(kMin, api::MakeValue(min)); }

float Clip::get_min() const {
  auto value_ptr = this->GetAttr(kMin);
  return GetValue<float>(value_ptr);
}

class ClipInferBase : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                             prim->name());
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                             prim->name());
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Clip, prim::kPrimClip, ClipInferBase, false);
}  // namespace ops
}  // namespace mindspore
