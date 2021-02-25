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

#include "ops/grad/flatten_grad.h"

namespace mindspore {
namespace ops {
AbstractBasePtr FlattenGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_x = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_shape = input_args[1]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto out_shape = GetValue<std::vector<int64_t>>(input_shape->BuildValue());
  auto ret = input_x->Broaden();
  ret->set_shape(std::make_shared<abstract::Shape>(out_shape));
  return ret;
}
REGISTER_PRIMITIVE_C(kNameFlattenGrad, FlattenGrad);
}  // namespace ops
}  // namespace mindspore
