/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_ACTION_INTERFACE_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_ACTION_INTERFACE_H_

#include "include/utils/visible.h"
#include "frontend/jit/ps/resource.h"

namespace mindspore {
namespace pipeline {
FRONTEND_EXPORT bool TaskEmitAction(const ResourcePtr &resource);
FRONTEND_EXPORT bool ExecuteAction(const ResourcePtr &resource);
FRONTEND_EXPORT FuncGraphPtr Renormalize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_ACTION_INTERFACE_H_
