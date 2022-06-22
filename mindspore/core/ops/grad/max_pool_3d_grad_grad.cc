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

#include "ops/grad/max_pool_3d_grad_grad.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr MaxPool3DGradGradInfer(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  return MaxPoolGradGradInfer(engine, primitive, input_args);
}

MIND_API_OPERATOR_IMPL(MaxPool3DGradGrad, MaxPoolGradGrad);
REGISTER_PRIMITIVE_EVAL_IMPL(MaxPool3DGradGrad, prim::kPrimMaxPool3DGradGrad, MaxPool3DGradGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
