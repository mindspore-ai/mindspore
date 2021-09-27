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

#include "backend/optimizer/pass/add_dynamic_shape_attr.h"
#include "ir/anf.h"
#include "utils/convert_utils.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {

const AnfNodePtr AddDynamicShapeAttr::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::IsNodeDynamicShape(node)) {
    AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), node);
    MS_LOG(INFO) << "Set Dynamic Shape Attr to Node:" << node->fullname_with_scope();
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    kernel_graph->SetGraphDynamicAttr(true);
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
