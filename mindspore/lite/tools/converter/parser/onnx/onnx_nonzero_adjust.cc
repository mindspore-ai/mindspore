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

#include "tools/converter/parser/onnx/onnx_nonzero_adjust.h"
#include "mindspore/core/ops/math_ops.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
bool OnnxNonZeroAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimWhere)) {
      continue;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_RET(prim != nullptr, false);
    if (prim->GetAttr("is_nonzero") == nullptr) {  // where operator does not have this attribute
      continue;
    }
    bool is_nonzero = GetValue<bool>(prim->GetAttr("is_nonzero"));
    if (!is_nonzero) {
      continue;
    }
    auto transpose = opt::GenTransposeNode(func_graph, cnode, {1, 0}, cnode->fullname_with_scope() + "_post");
    if (transpose == nullptr) {
      MS_LOG(ERROR) << "create transpose failed.";
      return false;
    }

    auto manager = Manage(func_graph, true);
    if (manager == nullptr) {
      MS_LOG(ERROR) << "manager is nullptr.";
      return false;
    }
    manager->Replace(cnode, transpose);
  }
  return true;
}
}  // namespace mindspore::lite
