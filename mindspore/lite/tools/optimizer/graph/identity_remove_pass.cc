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
#include "tools/optimizer/graph/identity_remove_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
bool RemoveIdentityOpPass::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type != lite::converter::FmkType_ONNX) {
    MS_LOG(INFO) << "The framework type of model should be onnx.";
    return RET_OK;
  }
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto type = opt::GetCNodeType(node);
    if (type != schema::PrimitiveType_Identity) {
      continue;
    }
    auto identity_cnode = node->cast<CNodePtr>();
    if (identity_cnode->inputs().size() != lite::kDoubleNum) {
      MS_LOG(ERROR) << "The `node input is a single tensor";
      return RET_ERROR;
    }
    manager->Replace(node, identity_cnode->input(1));
  }
  return true;
}
}  // namespace mindspore::opt
