/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/operator/ops_backend_infer_function.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/image_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/ops/primitive_infer_map.h"
namespace mindspore {
namespace abstract {
using R = PrimitiveEvalImplMap::mapped_type;
static PrimitiveEvalImplMap prim_backend_eval_implement_map{};
PrimitiveEvalImplMap *GetBackendPrimitiveInferMapPtr() { return &prim_backend_eval_implement_map; }
const PrimitiveEvalImplMap &GetBackendPrimitiveInferMap() { return prim_backend_eval_implement_map; }

std::optional<StandardPrimitiveImplReg> GetBackendPrimitiveInferImpl(const PrimitivePtr &primitive) {
  auto iter = GetBackendPrimitiveInferMap().find(primitive);
  if (iter != GetBackendPrimitiveInferMap().end()) {
    return iter->second;
  }

  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (found.has_value()) {
    return found.value();
  }
  return std::optional<StandardPrimitiveImplReg>();
}
}  // namespace abstract
}  // namespace mindspore
