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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ISOLATE_INPLACE_FUNC_REPLACE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ISOLATE_INPLACE_FUNC_REPLACE_H_

#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {

bool IsolateInplaceFuncReplace(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ISOLATE_INPLACE_FUNC_REPLACE_H_
