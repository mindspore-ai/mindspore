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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_BPROP_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_BPROP_MANAGER_H_

#include "include/common/utils/primitive_utils.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace ad {
#ifndef _WIN32
// For the bprop mindir generator.
// Given a python primitive or string, export a mindir file from the bprop defined in python.
void ExportBpropToMindir(const py::object &obj);
#endif
// Get bprop function of a primitive.
FuncGraphPtr GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources = nullptr);
}  // namespace ad
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_BPROP_MANAGER_H_
