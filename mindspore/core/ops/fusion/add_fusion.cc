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

#include "ops/fusion/add_fusion.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>

#include "ops/op_utils.h"
#include "utils/infer_base.h"

namespace mindspore {
namespace ops {
void AddFusion::set_activation_type(const ActivationType activation_type) {
  int64_t swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}
ActivationType AddFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void AddFusion::Init(const ActivationType activation_type) { this->set_activation_type(activation_type); }

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto add_prim = primitive->cast<PrimAddFusionPtr>();
  MS_EXCEPTION_IF_NULL(add_prim);
  auto op_name = add_prim->name();
  return BroadCastInferShape(op_name, input_args);
}
}  // namespace

AbstractBasePtr AddFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  size_t input_num = 2;
  auto type = InferBase::CheckSameInferType(primitive, input_args, common_valid_types, input_num);
  return std::make_shared<abstract::AbstractTensor>(type, InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameAddFusion, AddFusion);
}  // namespace ops
}  // namespace mindspore
