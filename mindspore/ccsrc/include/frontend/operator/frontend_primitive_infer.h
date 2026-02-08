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
#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATE_FRONTEND_PRIMITIVE_INFER_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATE_FRONTEND_PRIMITIVE_INFER_H_

#include <string>
#include <vector>
#include <optional>
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace abstract {
FRONTEND_EXPORT const PrimitiveEvalImplMap &GetFrontendPrimitiveInferMap();
// get prim infer from core/ops infer map or frontend infer map
FRONTEND_EXPORT std::optional<StandardPrimitiveImplReg> GetFrontendPrimitiveInferImpl(const PrimitivePtr &primitive);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATE_FRONTEND_PRIMITIVE_INFER_H_
