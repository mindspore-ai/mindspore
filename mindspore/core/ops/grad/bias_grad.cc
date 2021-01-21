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

#include "ops/grad/bias_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr BiasGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto bias_prim = primitive->cast<PrimBiasGradPtr>();
  MS_EXCEPTION_IF_NULL(bias_prim);
  auto prim_name = bias_prim->name();
  CheckAndConvertUtils::CheckInteger("bias_grad_infer", input_args.size(), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);

  // Infer shape
  auto inshape = CheckAndConvertUtils::ConvertShapePtrToShape("inshape", input_args[0]->BuildShape(), prim_name);
  for (size_t i = 0; i < inshape.size() - 1; i++) {
    inshape[i] = 1;
  }

  // Infer type
  auto intype = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();

  return std::make_shared<abstract::AbstractTensor>(intype, inshape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(BiasGrad, prim::kPrimBiasGrad, BiasGradInfer);
REGISTER_PRIMITIVE_C(kNameBiasGrad, BiasGrad);
}  // namespace ops
}  // namespace mindspore
