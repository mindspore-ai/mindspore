/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/add_dynamic_shape_attr.h"
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool AddDynamicShapeAttr::Process(const AnfNodePtr &node) const {
  if (common::AnfAlgo::IsDynamicShape(node)) {
    auto func_graph = node->func_graph();
    MS_LOG(DEBUG) << "Set Dynamic Shape Attr to Node:" << node->fullname_with_scope();
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    kernel_graph->SetGraphDynamicAttr(true);
    return true;
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
