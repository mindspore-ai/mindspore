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
#ifndef MINDSPORE_CCSRC_BACKEND_OPERATE_OPS_BACKEND_INFER_FUNCTION_H_
#define MINDSPORE_CCSRC_BACKEND_OPERATE_OPS_BACKEND_INFER_FUNCTION_H_
#include <optional>
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/backend/visible.h"
namespace mindspore {
namespace abstract {
BACKEND_EXPORT const PrimitiveEvalImplMap &GetBackendPrimitiveInferMap();
BACKEND_EXPORT PrimitiveEvalImplMap *GetBackendPrimitiveInferMapPtr();
// get prim infer from core/ops infer map or backend infer map
BACKEND_EXPORT std::optional<StandardPrimitiveImplReg> GetBackendPrimitiveInferImpl(const PrimitivePtr &primitive);
#define REGISTER_PRIMITIVE_BACKEND_EVAL_IMPL(name, primitive, infer_impl, infer_value_impl)                      \
  auto helper_##name = abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetBackendPrimitiveInferMapPtr(), \
                                                                     primitive, infer_impl, infer_value_impl, false);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATE_OPS_FRONT_INFER_FUNCTION_H_
