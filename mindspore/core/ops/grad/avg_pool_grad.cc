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

#include "ops/grad/avg_pool_grad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr AvgPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto AvgPoolGrad_prim = primitive->cast<PrimAvgPoolGradPtr>();
  MS_EXCEPTION_IF_NULL(AvgPoolGrad_prim);
  MS_EXCEPTION_IF_NULL(input_args[0]->BuildValue());
  auto origin_input_shape = GetValue<std::vector<int64_t>>(input_args[0]->BuildValue());
  auto tensor_type = input_args[1]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  return std::make_shared<abstract::AbstractTensor>(element, origin_input_shape);
}
REGISTER_PRIMITIVE_C(kNameAvgPoolGrad, AvgPoolGrad);
}  // namespace ops
}  // namespace mindspore
