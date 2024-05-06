/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_concat_adjust.h"
#include <string>
#include <vector>
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/array_ops.h"

namespace mindspore::lite {
namespace {
constexpr uint32_t kTwoNum = 2;
}  // namespace

bool OnnxConcatAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimConcat) || cnode->size() != kTwoNum) {
      continue;
    }
    MS_LOG(INFO) << "Del Concat node, node name: " << cnode->cast<CNodePtr>()->fullname_with_scope()
                 << ", node size: " << cnode->size();
    auto manager = Manage(func_graph);
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    (void)manager->Replace(cnode, cnode->cast<CNodePtr>()->input(1));
  }
  return true;
}
}  // namespace mindspore::lite
